import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torchmetrics
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import statistics
import Configuration
from functions.nn import NeuralNetwork
from functions.preprocess import preprocess_test
from PAL.Agent import Agent
from functions.clustering_test import Clustering
from sklearn import metrics


# Script to test the neural network using weights pre-trained by train.py (ONLINE)


# Set random seed
np.random.seed(0)
random.seed(0)

# Set environment path variables (for x server communication from WSL2 to Windows GUI)
os.environ['DISPLAY'] = "{}:0.0".format(Configuration.IP_ADDRESS)
os.environ['LIBGL_ALWAYS_INDIRECT'] = "0"

# Get cpu or gpu device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Select validation scenes
kitchens = [f"FloorPlan{i}" for i in range(20, 26)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(20, 26)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(20, 26)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(20, 26)]
val_scenes =  kitchens + living_rooms + bedrooms + bathrooms

# Select testing scenes
kitchens = [f"FloorPlan{i}" for i in range(26, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(26, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(26, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(26, 31)]
test_scenes = kitchens + living_rooms + bedrooms + bathrooms

# Set parameters
test_iter = 5
batch_size = 128
shuffle = True
learning_rate = 1e-4

epochs = 30
max_len = 30
soglia = 0.3


# Import neural network (create the same network and import weights)
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('nn/final_nn_ALL.pth', map_location=torch.device('cpu')))
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize dataframe for metrics
metrics_df = pd.DataFrame(columns=["Scene", "RI", "adjRI", "AMI", "NMI", "homogeneity", "completeness", "v_measure"])

for scene in val_scenes:

    # Initialize metrics and open tensorboard
    writer = SummaryWriter(log_dir='runs/Cluster_test_{}'.format(scene))
    test_accuracy = torchmetrics.Accuracy()
    test_precision = torchmetrics.Precision()
    test_recall = torchmetrics.Recall()

    # -------------------------------------------------------------------------------------------
    # Exploration phase - fine tuning of pre-trained model
    # -------------------------------------------------------------------------------------------
    print("Exploration phase")

    with open('preprocessed_dataset/prep_dataset_{}.pkl'.format(scene), "rb") as input_file:
        exploration_dataframe = pickle.load(input_file)

    with open('dataset/dataset_whole.pkl', "rb") as input_file:
        old_train_df = pickle.load(input_file)

    # Merge datasets
    exploration_dataframe = pd.concat([exploration_dataframe, old_train_df], ignore_index=True)

    # Separate target variable from rest of dataframe
    target = pd.DataFrame(exploration_dataframe.iloc[:, -1])
    dataframe = exploration_dataframe.drop(columns=exploration_dataframe.columns[-1], axis=1)

    # Load dataset in pytorch
    train = data_utils.TensorDataset(torch.Tensor(np.array(dataframe)), torch.Tensor(np.array(target)))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    size_train = len(train)

    # Train for n epochs on dataset created from exploration
    for epoch in range(epochs):

        # Switch model to training mode
        model.train()

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # -------------------------------------------------------------------------------------------
    # Moved objects phase - recognize objects and build clusters at each step
    # -------------------------------------------------------------------------------------------
    print("Testing phase")

    # Switch model to evaluation mode
    model.eval()

    # Initialize counters
    iterations = 0
    total = 0
    corrects = 0
    new_cluster_error = 0
    new_cluster_total = 0
    correct_types = 0
    performance = []

    # Initialize clustering class with clusters produced by exploration
    predicted_clustering = Clustering()

    with open('dataset/clustering_{}.pkl'.format(scene), "rb") as input_file:
        initial_clustering = pickle.load(input_file)
    # Calculate bb dimensions (columns 9, 10, 11, 12) and scale them
    initial_clustering.loc[:, 0] = (initial_clustering.iloc[:, 11] - initial_clustering.iloc[:, 9]) / 300
    initial_clustering.loc[:, 1] = (initial_clustering.iloc[:, 12] - initial_clustering.iloc[:, 10]) / 300
    initial_clustering.iloc[:, 8] = initial_clustering.iloc[:, 8] / 5
    # Drop the positions (0-5), angulations (6-7) and used bbs (9-12)
    cut_clustering = initial_clustering.drop(columns=[2, 3, 4, 5, 6, 7, 9, 10, 11, 12])
    cut_clustering.columns = list(range(np.shape(cut_clustering)[1]))
    converted_clustering = cut_clustering.to_numpy().tolist()
    # Add observations to clustering
    for element in converted_clustering:
        cluster_id = element[-1]
        obs = element[:-1]
        predicted_clustering.add_observation(obs, cluster_id)

    # Reduce clusters size to max_len
    predicted_clustering.update_clusters(max_len)

    # Generator to execute iterations
    generator = Agent(scene).run_modified(n_iter=test_iter)
    for iteration_output in generator:

        # Reset metrics calculators to prepare for new iteration
        test_accuracy.reset()
        test_precision.reset()
        test_recall.reset()

        # Cycle over the observation contained in the current iteration output to classify each of them
        for observation in iteration_output:

            # Update clusters
            predicted_clustering.update_clusters(max_len)
            predicted_clustering.update_list()
            past_obs = predicted_clustering.clustering_list

            # Vector to store the means for each cluster comparison with the new observation
            vector_means = np.array([])

            # Cycle over clusters
            for old_cluster in past_obs:

                # Convert to dataframe
                old_dataframe = pd.DataFrame(old_cluster)
                # Preprocess the dataframe to generate couples with current observation
                new_dataframe = preprocess_test(observation, old_dataframe)
                # Separate target variable from rest of dataframe
                target = pd.DataFrame(new_dataframe.iloc[:, -1])
                tf_dataframe = new_dataframe.drop(columns=new_dataframe.columns[-1], axis=1)
                # Load dataset in pytorch
                test = data_utils.TensorDataset(torch.Tensor(np.array(tf_dataframe)), torch.Tensor(np.array(target)))
                test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

                # List to store the outputs
                output = []

                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels = data
                        size = labels.size(0)
                        outputs = model(inputs)

                        # Store the outputs
                        output += outputs.flatten().tolist()

                        # Update metrics with torchmetrics
                        test_accuracy.update(outputs, labels.int())
                        test_precision.update(outputs, labels.int())
                        test_recall.update(outputs, labels.int())

                # Calculate list of n max elements
                output_n_max = np.argsort(output)[-5:]
                n_max = []
                for index in output_n_max:
                    n_max.append(output[index])

                # Calculate the mean of the outputs and store it (or mean of n maximums)
                # Strategy 1: mean
                mean = statistics.mean(output)
                vector_means = np.append(vector_means, mean)

                # Strategy 2: 3rd highest element
                #max1 = np.max(output)
                #output.remove(max1)
                #max2 = np.max(output)
                #output.remove(max2)
                #max3 = np.max(output)
                #vector_means = np.append(vector_means, max3)
                #vector_means = np.append(vector_means, n_max[2])

                # Strategy 3: highest element
                #max = np.max(output)
                #vector_means = np.append(vector_means, max)

                # Strategy 4: mean of n max
                #mean = statistics.mean(n_max)
                #vector_means = np.append(vector_means, mean)


            # Decide to which cluster assign this observation
            assign_index = np.where(vector_means == np.max(vector_means))
            # Check for multiple indexes
            assign_index = int(assign_index[0])

            if np.max(vector_means) < soglia:
                # Create new cluster
                new_index = len(predicted_clustering.clusters.keys())
                predicted_clustering.initialize(new_index, observation)
                new_cluster_total += 1
                # Check correctness
                id = observation[-1]
                for i in predicted_clustering.clusters.keys():
                    old_id = predicted_clustering.clusters[i][0][-1]
                    if id == old_id:
                        new_cluster_error += 1
                        print("Wrong cluster created")

            else:
                # Add observation to the clustering
                predicted_clustering.add_observation(observation, assign_index)

                # Check if the prediction is correct
                pred_label = predicted_clustering.clusters[assign_index][0][-1]
                true_label = observation[-1]
                total += 1
                correct = pred_label == true_label
                corrects += correct

                pred_type = pred_label.partition("|")[0]
                true_tpye = true_label.partition("|")[0]
                correct_type = pred_type == true_tpye
                correct_types += correct_type
                #print("Pred:",pred_label,"- True:", true_label)


        # Compute metrics for the current iteration
        acc = test_accuracy.compute()
        pre = test_precision.compute()
        rec = test_recall.compute()
        # Add values to tensorboard
        writer.add_scalar("Test/Accuracy", acc.item(), iterations)
        writer.add_scalar("Test/Precision", pre.item(), iterations)
        writer.add_scalar("Test/Recall", rec.item(), iterations)
        # Store pred
        performance.append([total, corrects, correct_types, iterations])
        # Update iteration count
        iterations += 1

    # Close tensorboard
    writer.flush()
    writer.close()

    # Write performance to file
    df_performance = pd.DataFrame(performance)
    name = "performance/performance_{}_{}_{}".format(scene, epochs, max_len)
    with open(name, 'wb') as outfile:
        pickle.dump(performance, outfile)
    df_performance.to_excel(name+".xlsx")

    print("Observation inserted in clusters:", total)
    print("Observation inserted in clusters correctly:", corrects)
    print("Observation inserted in clusters with correct type:", correct_types)
    print("New clusters created:", new_cluster_total)
    print("New clusters created wrong:", new_cluster_error)

    # Calculate metrics for the clustering
    labels_true = []
    labels_pred = []
    # Cycle over clusters
    for index in predicted_clustering.clusters.keys():
        cluster = predicted_clustering.clusters[index]
        true_id = cluster[0][-1]
        # Cycle over observations of each cluster and append true/predicted label to the lists
        for element in cluster:
            labels_true.append(true_id)
            labels_pred.append(element[-1])
    # Calculates metrics
    rand = metrics.rand_score(labels_true, labels_pred)
    adj_rand = metrics.adjusted_rand_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_pred, labels_true)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    completeness = metrics.completeness_score(labels_true, labels_pred)
    v_measure = metrics.v_measure_score(labels_true, labels_pred)

    # Store metrics
    series = pd.Series({"Scene": scene, "RI": rand, "adjRI": adj_rand, "AMI": AMI, "NMI": NMI,
                        "homogeneity": homogeneity, "completeness": completeness, "v_measure":v_measure})
    metrics_df.append(series)
    metrics_df.to_excel("Metrics.xlsx")



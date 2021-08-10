import random
import numpy as np
import pickle
from functions.nn import NeuralNetwork
import torch
from torch import nn
import pandas as pd
import torch.utils.data as data_utils
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# Script to train the neural network using datasets from files

# Set random seed
np.random.seed(0)
random.seed(0)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Set Neural Network parameters, Create an instance of NeuralNetwork, and move it to the device
learning_rate = 1e-4
#base_lr = 1e-5
epochs = 60
batch_size = 128
shuffle = True
n_layers = 4
model = NeuralNetwork().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, learning_rate, cycle_momentum=False)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Open tensorboard
writer = SummaryWriter(
    log_dir='runs/TEST FINAL 2 - Lr{} - Batch{} - {}layers - leakyrelu'.format(learning_rate, batch_size, n_layers))

# Print the model
summary(model, input_size=(1, 1030))

# Detect anomaly
# torch.autograd.set_detect_anomaly(True)


# -------------------------------------------------------------------------------------------------
# Prepare the datasets with all scenes
# -------------------------------------------------------------------------------------------------

# Select training scenes
kitchens = [f"FloorPlan{i}" for i in range(1, 20)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 20)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 20)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 20)]
train_scenes = kitchens + living_rooms + bedrooms + bathrooms

# Select validation scenes
kitchens = [f"FloorPlan{i}" for i in range(20, 26)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(20, 26)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(20, 26)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(20, 26)]
val_scenes = kitchens + living_rooms + bedrooms + bathrooms

# Select testing scenes
kitchens = [f"FloorPlan{i}" for i in range(26, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(26, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(26, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(26, 31)]
test_scenes = kitchens + living_rooms + bedrooms + bathrooms

# Create dataset with all scenes dataframes
all_scenes = pd.DataFrame()
for scene in train_scenes:
    with open('preprocessed_dataset2/prep_dataset_{}.pkl'.format(scene), "rb") as input_file:
        preprocessed_dataframe = pickle.load(input_file)
    all_scenes = pd.concat([all_scenes, preprocessed_dataframe], ignore_index=True)
    del preprocessed_dataframe

# Separate target variable from rest of dataframe
target = pd.DataFrame(all_scenes.iloc[:, -1])
dataframe = all_scenes.drop(columns=all_scenes.columns[-1], axis=1)

# Load dataset in pytorch
train = data_utils.TensorDataset(torch.Tensor(np.array(dataframe)), torch.Tensor(np.array(target)))
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)
size_train = len(train)
print(target.value_counts())

print("Training set loaded")

# Create dataset with all validation or testing scenes dataframes
all_scenes_test = pd.DataFrame()
for scene in test_scenes:
    with open("preprocessed_dataset2/prep_dataset_{}.pkl".format(scene), "rb") as input_file:
        preprocessed_dataframe = pickle.load(input_file)
    all_scenes_test = pd.concat([all_scenes_test, preprocessed_dataframe], ignore_index=True)
    del preprocessed_dataframe

# Separate target variable from rest of dataframe
test_target = pd.DataFrame(all_scenes_test.iloc[:, -1])
test_dataframe = all_scenes_test.drop(columns=all_scenes_test.columns[-1], axis=1)
print(test_target.value_counts())

# Load dataset in pytorch
test = data_utils.TensorDataset(torch.Tensor(np.array(test_dataframe)), torch.Tensor(np.array(test_target)))
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=shuffle)
size_test = len(test)

# print(test_target.value_counts())
print("Test set loaded")

# -------------------------------------------------------------------------------------------------
# Initialize metrics
# -------------------------------------------------------------------------------------------------

# Initialize metrics with torchmetrics
train_accuracy = torchmetrics.Accuracy().to(device)
train_precision = torchmetrics.Precision().to(device)
train_recall = torchmetrics.Recall().to(device)
test_accuracy = torchmetrics.Accuracy().to(device)
test_precision = torchmetrics.Precision().to(device)
test_recall = torchmetrics.Recall().to(device)

# -------------------------------------------------------------------------------------------------
# Train network for n epochs
# -------------------------------------------------------------------------------------------------
for epoch in range(epochs):

    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc
    # which behave differently in training and evaluation mode
    model.train()
    print("Training Epoch ", epoch + 1)

    # Reset metrics calculators to prepare for new epoch
    running_loss = 0.0
    train_accuracy.reset()
    train_precision.reset()
    train_recall.reset()

    # Train
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

        # Update metrics
        train_accuracy.update(outputs, labels.int())
        train_precision.update(outputs, labels.int())
        train_recall.update(outputs, labels.int())
        running_loss += loss.item() * inputs.size(0)



    # Add weight graphs to tensorboard
    writer.add_graph(model, inputs)
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, epoch)
#        writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    # Compute metrics for the current epoch
    acc = train_accuracy.compute()
    pre = train_precision.compute()
    rec = train_recall.compute()
    epoch_loss = running_loss / size_train

    # Add values to tensorboard
    writer.add_scalar("Train/Accuracy", acc.item(), epoch)
    writer.add_scalar("Train/Precision", pre.item(), epoch)
    writer.add_scalar("Train/Recall", rec.item(), epoch)
    writer.add_scalar("Train/Loss", epoch_loss, epoch)

    # -------------------------------------------------------------------------------------------
    # Testing phase
    # -------------------------------------------------------------------------------------------

    # Switch model to evaluation mode.
    model.eval()
    print("Testing...")

    # Reset metrics calculators to prepare for new iteration
    running_loss = 0.0
    test_accuracy.reset()
    test_precision.reset()
    test_recall.reset()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            size = labels.size(0)
            outputs = model(images)
            predicted = outputs > 0.5  # tensor of true/false
            loss = loss_fn(outputs, labels)

            # Update metrics
            test_accuracy.update(outputs, labels.int())
            test_precision.update(outputs, labels.int())
            test_recall.update(outputs, labels.int())
            running_loss += loss.item() * images.size(0)

        # Compute and store metrics in lists for the current iteration
        acc = test_accuracy.compute()
        pre = test_precision.compute()
        rec = test_recall.compute()
        epoch_loss = running_loss / size_test

        # Add values to tensorboard
        writer.add_scalar("Test/Accuracy", acc.item(), epoch)
        writer.add_scalar("Test/Precision", pre.item(), epoch)
        writer.add_scalar("Test/Recall", rec.item(), epoch)
        writer.add_scalar("Test/Loss", epoch_loss, epoch)

    # Scheduler step
    #scheduler.step()



# Record hyper-parameters (with final metrics)
writer.add_hparams(hparam_dict={"Learning Rate": learning_rate, "Batch Size": batch_size, "Shuffle": shuffle,
                                "Num Layers": n_layers},
                   metric_dict={"accuracy": acc, "precision": pre, "recall": rec, "loss": epoch_loss})

# Save final model
PATH = './nn/final_nn_ALL.pth'
torch.save(model.state_dict(), PATH)

# Close tensorboard
writer.flush()
writer.close()

import random
import numpy as np
import pickle
from functions.preprocess import preprocess, preprocess_all

# Script to create dataframe with couples of observations from datasets in dataset folder

# Set random seed
np.random.seed(0)
random.seed(0)


# Select training scenes
kitchens = [f"FloorPlan{i}" for i in range(1, 26)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 26)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 26)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 26)]
train_scenes = kitchens + living_rooms + bedrooms + bathrooms

kitchens = [f"FloorPlan{i}" for i in range(26, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(26, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(26, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(26, 31)]
test_scenes = kitchens + living_rooms + bedrooms + bathrooms

all_scenes = train_scenes + test_scenes

# Set algorithm parameters
max_selected_obs = 200
num_ex = 10


# Iterate over scenes
for scene in all_scenes:

        # Import data from files
        with open("dataset/clustering_{}.pkl".format(scene), "rb") as input_file:
            initial_clustering = pickle.load(input_file)
        print(scene)
        print("Data original shape: ", np.shape(initial_clustering))

        # Prepare dataset for neural network

        # Calculate bb dimensions (columns 9, 10, 11, 12) and scale them
        initial_clustering.loc[:, 0] = (initial_clustering.iloc[:, 11] - initial_clustering.iloc[:, 9])/300
        initial_clustering.loc[:, 1] = (initial_clustering.iloc[:, 12] - initial_clustering.iloc[:, 10])/300
        initial_clustering.iloc[:, 8] = initial_clustering.iloc[:, 8]/5

        # Drop the positions (0-5), angulations (6-7) and used bbs (9-12).
        # Keep 0, 1 where we stored bb dimensions and keep 8 which is distance
        cut_clustering = initial_clustering.drop(columns=[2, 3, 4, 5, 6, 7, 9, 10, 11, 12])

        # Rename the columns to adjust for dropping
        cut_clustering.columns = list(range(np.shape(cut_clustering)[1]))


        # Form the couples
        preprocessed_dataframe = preprocess(cut_clustering, max_selected_obs=max_selected_obs, num_ex=num_ex)

        print("Preprocessed dataset shape: ", np.shape(preprocessed_dataframe))
        print("Number of examples: ", preprocessed_dataframe.iloc[:,-1].value_counts())
        print("\n")

        # Scale features
        #scaled_dataframe = pd.DataFrame(MinMaxScaler().fit_transform(preprocessed_dataframe))

        # Save preprocessed datasets
        preprocessed_dataframe.to_pickle("preprocessed_dataset/prep_dataset_{}.pkl".format(scene))


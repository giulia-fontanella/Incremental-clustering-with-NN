import pickle
import pandas as pd


# Select training or testing scenes
#kitchens = [f"FloorPlan{i}" for i in range(26, 30)]
#living_rooms = [f"FloorPlan{200 + i}" for i in range(26, 30)]
#bedrooms = [f"FloorPlan{300 + i}" for i in range(26, 30)]
#bathrooms = [f"FloorPlan{400 + i}" for i in range(26, 30)]
#test_scenes = kitchens + living_rooms + bedrooms + bathrooms

kitchens = [f"FloorPlan{i}" for i in range(1, 20)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 20)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 20)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 20)]
train_scenes = kitchens + living_rooms + bedrooms + bathrooms

# Create dataset with all kitchen scenes and save it
kitchen_df = pd.DataFrame()
for scene in kitchens:
    with open("preprocessed_dataset/prep_dataset_{}.pkl".format(scene), "rb") as input_file:
        preprocessed_dataframe = pickle.load(input_file)
    kitchen_df = pd.concat([kitchen_df, preprocessed_dataframe], ignore_index=True)
    del preprocessed_dataframe
with open("dataset/dataset_whole/dataset_kitchens.pkl", 'wb') as outfile:
    pickle.dump(kitchen_df, outfile)
del kitchen_df


# Create dataset with all living room scenes and save it
living_rooms_df = pd.DataFrame()
for scene in living_rooms:
    with open("preprocessed_dataset/prep_dataset_{}.pkl".format(scene), "rb") as input_file:
        preprocessed_dataframe = pickle.load(input_file)
    living_rooms_df = pd.concat([living_rooms_df, preprocessed_dataframe], ignore_index=True)
    del preprocessed_dataframe
with open("dataset/dataset_whole/dataset_living_rooms.pkl", 'wb') as outfile:
    pickle.dump(living_rooms_df, outfile)
del living_rooms_df

# Create dataset with all bedroom scenes and save it
bedrooms_df = pd.DataFrame()
for scene in bedrooms:
    with open("preprocessed_dataset/prep_dataset_{}.pkl".format(scene), "rb") as input_file:
        preprocessed_dataframe = pickle.load(input_file)
    bedrooms_df = pd.concat([bedrooms_df, preprocessed_dataframe], ignore_index=True)
    del preprocessed_dataframe
with open("dataset/dataset_whole/dataset_bedrooms.pkl", 'wb') as outfile:
    pickle.dump(bedrooms_df, outfile)
del bedrooms_df

# Create dataset with all bathroom scenes and save it
bathrooms_df = pd.DataFrame()
for scene in bathrooms:
    with open("preprocessed_dataset/prep_dataset_{}.pkl".format(scene), "rb") as input_file:
        preprocessed_dataframe = pickle.load(input_file)
    bathrooms_df = pd.concat([bathrooms_df, preprocessed_dataframe], ignore_index=True)
    del preprocessed_dataframe
with open("dataset/dataset_whole/dataset_bathrooms.pkl", 'wb') as outfile:
    pickle.dump(bathrooms_df, outfile)
del bathrooms_df


# Concatenate all the datasets
with open("dataset/dataset_whole/dataset_kitchens.pkl", 'rb') as file1, open("dataset/dataset_whole/dataset_living_rooms.pkl", 'rb') as file2, \
        open("dataset/dataset_whole/dataset_living_rooms.pkl", 'rb') as file3, open("dataset/dataset_whole/dataset_living_rooms.pkl", 'rb') as file4:
                df1 = pickle.load(file1)
                df2 = pickle.load(file2)
                df3 = pickle.load(file3)
                df4 = pickle.load(file4)
                df = pd.concat([df1, df2, df3, df4], ignore_index=True)
with open("dataset/dataset_whole/dataset_whole.pkl", 'wb') as outfile:
    pickle.dump(df, outfile)



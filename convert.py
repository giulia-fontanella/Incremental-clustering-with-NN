import pickle
import pickle4

# Select training scenes
kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]
all_scenes = kitchens + living_rooms + bedrooms + bathrooms

for scene in all_scenes:
    with open("preprocessed_dataset/prep_dataset_{}.pkl".format(scene), "rb") as input:
        file = pickle.load(input)

    with open("preprocessed_dataset2/prep_dataset_{}.pkl".format(scene), "wb") as output:
        pickle.dump(file, output, 4)






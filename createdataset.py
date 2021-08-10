import os
import random
import numpy as np
import Configuration
from PAL.Agent import Agent
from functions.savedataset import savedataset

# Script to run Agent for n=50 iterations and save the resulting initial clustering in files

# Set random seed
np.random.seed(0)
random.seed(0)

# Set environment path variables (for x server communication from WSL2 to Windows GUI)
os.environ['DISPLAY'] = "{}:0.0".format(Configuration.IP_ADDRESS)
os.environ['LIBGL_ALWAYS_INDIRECT'] = "0"

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
all_scenes = ["FloorPlan8"]

for scene in all_scenes:

    print(scene)

    # Run the agent
    result = Agent(scene).run(n_iter=50)

    # Save datasets in files
    savedataset(result[0], result[1], scene, plot_obj_types=False)

    # Clear memory
    del result



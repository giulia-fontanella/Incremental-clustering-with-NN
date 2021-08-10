
import pandas as pd
import numpy as np
from PIL import Image
from PAL.resnet import FeatureExtractor


# Function used by run_modified. It gets in input a new state and gives
# as output a dataframe with all the observations associated to the state


class GetObs:

    def __init__(self):
        self.feat_ext = FeatureExtractor()


    def get_obs(self, state):
        new_obs = []
        pos_agent = state.perceptions[0:3]
        angles = state.perceptions[6:8]
        rgb = state.perceptions[8:270008]
        rgb_img = rgb.reshape((300, 300, 3))
        visible_objs = state.visible_objects
        bb = state.bb

        for obj_class in visible_objs.keys():
            for obj in visible_objs[obj_class]:
                name = obj["name"]
                x = obj['map_x']
                y = obj["map_y"]
                z = obj["map_z"]
                distance = np.sqrt((pos_agent [0] - x) ** 2 + (pos_agent [1] - y) ** 2 + (pos_agent [2] - z) ** 2)

                # Cut rgb image
                if name in bb.keys():
                    obj_bbox = bb[name]
                    rgb_cut = rgb_img[max(0, obj_bbox[1] - 3): min(rgb_img.shape[0], obj_bbox[3] + 4),
                              max(0, obj_bbox[0] - 3): min(rgb_img.shape[1], obj_bbox[2] + 4), :]

                    # Extract features
                    rgb_pil = Image.fromarray(np.uint8(rgb_cut)).convert('RGB')
                    rgb_features = self.feat_ext.extract_features(rgb_pil)

                    # Calculate bb dimensions
                    dim_bb1 = obj_bbox[2] - obj_bbox[0]
                    dim_bb2 = obj_bbox[3] - obj_bbox[1]

                    # Create list with observation data and append it to the list of new observations new_obs
                    observation = [dim_bb1, dim_bb2, distance]
                    observation.extend(rgb_features.tolist())
                    observation.append(name)
                    new_obs.append(observation)


        # Convert list to a dataframe
        new_obs_dataframe = pd.DataFrame()
        new_obs_dataframe = new_obs_dataframe.append(new_obs)

        return new_obs_dataframe


    def get_obs_modified(self, state):
        new_obs = []
        pos_agent = state.perceptions[0:3]
        angles = state.perceptions[6:8]
        rgb = state.perceptions[8:270008]
        rgb_img = rgb.reshape((300, 300, 3))
        visible_objs = state.visible_objects
        bb = state.bb

        for obj_class in visible_objs.keys():
            for obj in visible_objs[obj_class]:
                name = obj["name"]
                x = obj['map_x']
                y = obj["map_y"]
                z = obj["map_z"]
                distance = np.sqrt((pos_agent [0] - x) ** 2 + (pos_agent [1] - y) ** 2 + (pos_agent [2] - z) ** 2)

                # Cut rgb image
                if name in bb.keys():
                    obj_bbox = bb[name]
                    rgb_cut = rgb_img[max(0, obj_bbox[1] - 3): min(rgb_img.shape[0], obj_bbox[3] + 4),
                              max(0, obj_bbox[0] - 3): min(rgb_img.shape[1], obj_bbox[2] + 4), :]

                    # Extract features
                    rgb_pil = Image.fromarray(np.uint8(rgb_cut)).convert('RGB')
                    rgb_features = self.feat_ext.extract_features(rgb_pil)

                    # Calculate bb dimensions
                    dim_bb1 = obj_bbox[2] - obj_bbox[0]
                    dim_bb2 = obj_bbox[3] - obj_bbox[1]

                    # Create list with observation data and append it to the list of new observations new_obs
                    observation = [dim_bb1/300, dim_bb2/300, distance/5]
                    observation.extend(rgb_features.tolist())
                    observation.append(name)
                    new_obs.append(observation)

        return new_obs


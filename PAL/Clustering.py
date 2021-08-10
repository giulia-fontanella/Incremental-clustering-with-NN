import numpy as np
from PAL.Cluster import Cluster
import math
from PAL.resnet import FeatureExtractor
from PIL import Image

class Clustering:

    def __init__(self):

        # Initialize clusters (will be a list of Cluster objects)
        self.clusters = None
        self.feat_ext = FeatureExtractor()
        self.ID = 0

    def update_clusters(self, state):

        pos_agent = state.perceptions[0:3]
        angles = state.perceptions[6:8]

        rgb = state.perceptions[8:270008]
        rgb_img = rgb.reshape((300, 300, 3))
        visible_objs = state.visible_objects
        bb = state.bb
        old_clusters = self.clusters

        # Initialize clusters for the first time, one for each object
        if self.clusters is None:
            new = []
            for obj_class in visible_objs.keys():
                for obj in visible_objs[obj_class]:
                    name = obj["name"]
                    x = obj['map_x']
                    y = obj["map_y"]
                    z = obj["map_z"]
                    distance = np.sqrt((pos_agent[0]-x)**2 + (pos_agent[1]-y)**2 + (pos_agent[2]-z)**2)

                    if name in bb.keys():

                        # Cut rgb image
                        obj_bbox = bb[name]
                        rgb_cut = rgb_img[max(0, obj_bbox[1] - 3): min(rgb_img.shape[0], obj_bbox[3] + 4),
                                  max(0, obj_bbox[0] - 3): min(rgb_img.shape[1], obj_bbox[2] + 4), :]

                        # Rescale image and extract features
                        rgb_pil = Image.fromarray(np.uint8(rgb_cut)).convert('RGB')
                        rgb_features = self.feat_ext.extract_features(rgb_pil)

                        # Create array with observation data
                        pos = np.array([x, y, z])
                        obj_bb = np.array(obj_bbox)
                        distance = np.array([distance])
                        observation = np.concatenate([pos, pos_agent, angles, distance, obj_bb, rgb_features])

                        # Add a new cluster with the observation
                        new.append(Cluster(name, [observation], self.ID))
                        self.ID += 1

            self.clusters = new



        # Compare positions of the visible objects with old clusters and update them
        else:
            for obj_class in visible_objs.keys():
                for obj in visible_objs[obj_class]:
                    name = obj["name"]
                    x = obj['map_x']
                    y = obj["map_y"]
                    z = obj["map_z"]
                    distance = np.sqrt((pos_agent[0]-x)**2 + (pos_agent[1]-y)**2 + (pos_agent[2]-z)**2)

                    if name in bb.keys():
                        obj_bbox = bb[name]
                        rgb_cut = rgb_img[max(0, obj_bbox[1] - 3): min(rgb_img.shape[0], obj_bbox[3] + 4),
                                  max(0, obj_bbox[0] - 3): min(rgb_img.shape[1], obj_bbox[2] + 4), :]

                        # Rescale image and extract features
                        rgb_pil = Image.fromarray(np.uint8(rgb_cut)).convert('RGB')
                        rgb_features = self.feat_ext.extract_features(rgb_pil)

                        pos = np.array([x, y, z])
                        obj_bb = np.array(obj_bbox)
                        distance = np.array([distance])
                        new_obs = np.concatenate([pos, pos_agent, angles, distance, obj_bb, rgb_features])

                        # Variable to indicate if an object has been stored in a cluster
                        obj["status"] = 0

                        for clust in old_clusters:
                            # Compare position with the positions of objects in previous clusters
                            x2 = clust.observations[0][0]
                            y2 = clust.observations[0][1]
                            z2 = clust.observations[0][2]
                            eps = 0.01
                            if math.isclose(x, x2, rel_tol=eps) and math.isclose(y, y2, rel_tol=eps) and math.isclose(z,
                                                                                                                      z2,
                                                                                                                      rel_tol=eps):
                                clust.add_observation(new_obs)
                                obj["status"] = 1

                        # If an object wasn't added to any cluster, create a new cluster
                        if obj["status"] == 0:
                            self.clusters.append(Cluster(name, [new_obs], self.ID))
                            self.ID += 1

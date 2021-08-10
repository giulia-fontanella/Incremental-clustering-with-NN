import os
from collections import defaultdict
import numpy as np


class ObjectDetector:


    def __init__(self):
         pass


    def get_visible_objects_ground_truth(self, metadata):
        visible_objects = defaultdict(list)

        for obj in metadata['objects']:
            if obj['visible']:
                obj_id = "{}_{}".format(obj['objectType'].lower(), len(visible_objects[obj['objectType'].lower()]))
                obj_name = obj['objectId']
                obj_map_x = obj['position']['x']
                obj_map_y = obj['position']['z']
                obj_map_z = obj['position']['y']

                # Set table z coordinate to table height
                if obj_name.lower().startswith("diningtable"):
                    obj_map_z = np.max(np.array(obj['axisAlignedBoundingBox']['cornerPoints'])[:, 1])

                obj_bound_box = obj['axisAlignedBoundingBox']
                # obj_distance = obj['distance']
                visible_objects[obj['objectType'].lower()].append({"id":obj_id,
                                                                   "name": obj_name,# only for ground truth
                                                                   "map_x":obj_map_x,
                                                                   "map_y":obj_map_y,
                                                                   "map_z":obj_map_z,
                                                                   "bound_box":obj_bound_box
                                                           # "distance": obj_distance}
                                                            })

        return dict(visible_objects)









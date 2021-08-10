from collections import defaultdict

class KnowledgeManager:


    def __init__(self):
        self.all_objects = defaultdict(list)


    def update_objects(self, new_objects):
        position_threshold = 0.1 # if some x y z coordinate is above threshold, then the object instance is a new one

        objects_id_mapping = dict()

        # Add new objects to current objects dictionary
        for obj_type in new_objects.keys():
            # The object type is a new one
            if len(self.all_objects[obj_type]) == 0:
                self.all_objects[obj_type].extend(new_objects[obj_type])

                # Update objects id mapping
                for obj in new_objects[obj_type]:
                    objects_id_mapping[obj['id']] = {'id': obj['id'], 'name': obj['name']}

            # There are already some object instances of object type
            else:
                for new_obj_type_inst in new_objects[obj_type]:

                    # Store new object id
                    # objects_id_mapping[new_obj_type_inst['id']] = new_obj_type_inst['id']

                    new_obj_x, new_obj_y, new_obj_z = new_obj_type_inst['map_x'], new_obj_type_inst['map_y'], new_obj_type_inst['map_z']

                    new_obj_exists = False

                    # Check if an object instance does not exist yet in the current objects dictionary
                    for existing_obj in self.all_objects[obj_type]:
                        exist_obj_x, exist_obj_y, exist_obj_z = existing_obj['map_x'], existing_obj['map_y'], existing_obj['map_z']

                        if (new_obj_x - exist_obj_x) < position_threshold \
                            and (new_obj_y - exist_obj_y) < position_threshold \
                            and (new_obj_z - exist_obj_z) < position_threshold:
                            # Change (not) new object instance id to already existing one
                            objects_id_mapping[new_obj_type_inst['id']] = {'id': existing_obj['id'],
                                                                           'name':existing_obj['name']}
                            new_obj_exists = True
                            break

                    # If new object instance does not exist yet in the current objects dictionary
                    if not new_obj_exists:
                        # Update new object id
                        new_obj_id = "{}_{}".format(obj_type, len(self.all_objects[obj_type]))
                        objects_id_mapping[new_obj_type_inst['id']] = {'id':new_obj_id, 'name':new_obj_type_inst['name']}
                        # object instance is a new one
                        new_obj_type_inst['id'] = new_obj_id
                        self.all_objects[obj_type].append(new_obj_type_inst)

        # # Update visible objects id
        # for obj_id in objects_id_mapping.keys():
        #     new_objects[obj_id] = objects_id_mapping[obj_id]

        # Return visible objects with updated id
        return objects_id_mapping

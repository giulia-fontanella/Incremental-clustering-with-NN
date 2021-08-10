import copy
import os
from collections import defaultdict
import random
import numpy as np
from ai2thor.controller import Controller
from PAL.Clustering import Clustering
from functions.get_obs import GetObs
import Configuration
from PAL.Learn.Learner import Learner
from PAL.Learn.EnvironmentModels.State import State
from PAL.Plan.EventPlanner import EventPlanner
from Utils import Logger
import cv2

class Agent:


    def __init__(self,scene):

        # Set learner
        self.learner = Learner()

        # Set event planner
        self.event_planner = EventPlanner(self.learner.mapper.map_model)


        # Set simulator controller
        self.controller = Controller(renderDepthImage=Configuration.RENDER_DEPTH_IMG,
                                     renderObjectImage=True, visibilityDistance="5",
                                     gridSize=0.1, scene=scene)

        # Initialize event (i.e. the observation after action execution)
        self.event = None

        # Initialize current state
        self.state = self.observe()

        # Current iteration
        self.iter = 0

        # Update agent position in agent state and path planner state
        self.pos = {"x":int(), "y":int(), "z":int()}
        self.hand_pos = {"x":int(), "y":int(), "z":int()}
        self.angle = None
        self.update()

        # Initialize clusters
        self.clustering = Clustering()

        # Initialize class getobs
        self.GetObs = GetObs()


    def run(self, n_iter=Configuration.MAX_ITER):

        # Iterate for a maximum number of steps
        for i in range(n_iter):

            # Set current iteration number
            self.iter = i

            # Update top view through depth matrix if camera is aligned with the x axis
            if int(self.event.metadata['agent']['cameraHorizon']) == 0:
                self.learner.update_topview(os.path.join(Logger.LOG_DIR_PATH, "topview_{}.png".format(self.iter)),
                                            self.event.depth_frame, self.angle, self.pos)

            # Choose an event through event planner
            event_action = self.event_planner.plan()

            Logger.write('{}:{}'.format(self.iter + 1, event_action))

            # Execute the chosen action
            self.event = self.step(event_action)

            # Detect collision
            if event_action == "MoveAhead" and not self.event.metadata["lastActionSuccess"]:
                # self.update_collision_map(agent_orientation)
                self.update_collision_map(self.angle)
                self.event_planner.path_plan = None

            # Look if pddl action has been successfully executed and in such case update predicates
            elif self.event.metadata["lastActionSuccess"] and self.event_planner.event_plan is not None \
                    and len(self.event_planner.event_plan) == 0:

                # DEBUG
                print("Successfully executed action: {}".format(self.event_planner.subgoal))

            # Save agent view image
            if Configuration.PRINT_CAMERA_VIEW_IMAGES:
                Logger.save_img("view_{}.png".format(i), self.event.frame)

            # Save agent depth view image
            if Configuration.PRINT_CAMERA_DEPTH_VIEW_IMAGES:
                Logger.save_img("depth_view_{}.png".format(i), (self.event.depth_frame/np.max(self.event.depth_frame)*255).astype('uint8'))

            # Observe new state
            new_state = self.observe()

            # Add state in abstract model
            self.learner.add_state(new_state)

            # Add transition
            self.learner.add_transition(self.state, event_action, new_state)

            # Update current state
            self.state = new_state

            # Update agent position in agent state and path planner state
            self.update()

            # Update the clusters
            self.clustering.update_clusters(new_state)

        return [self.learner.abstract_model.states, self.clustering]


    def run_modified(self, n_iter=Configuration.MAX_ITER):

        # Move the objects randomly
        self.controller.step(action="InitialRandomSpawn", randomSeed=0, forceVisible=True,
                             numPlacementAttempts=5, placeStationary=True)

        # Iterate for a maximum number of steps
        for i in range(n_iter):

            # Set current iteration number
            self.iter = i

            # Update top view through depth matrix if camera is aligned with the x axis
            if int(self.event.metadata['agent']['cameraHorizon']) == 0:
                self.learner.update_topview(os.path.join(Logger.LOG_DIR_PATH, "topview_{}.png".format(self.iter)),
                                            self.event.depth_frame, self.angle, self.pos)

            # Choose an event through event planner
            event_action = self.event_planner.plan()

            Logger.write('{}:{}'.format(self.iter + 1, event_action))

            # Move objects randomly in the scene before executing next action
            #self.event = self.move_objs()
            # Set visibility distance back to normal
            #self.controller.reset(visibilityDistance="5")

            # Execute the chosen action
            self.event = self.step(event_action)

            # Detect collision
            if event_action == "MoveAhead" and not self.event.metadata["lastActionSuccess"]:
                # self.update_collision_map(agent_orientation)
                self.update_collision_map(self.angle)
                self.event_planner.path_plan = None

            # Look if pddl action has been successfully executed and in such case update predicates
            elif self.event.metadata["lastActionSuccess"] and self.event_planner.event_plan is not None \
                    and len(self.event_planner.event_plan) == 0:

                # DEBUG
                print("Successfully executed action: {}".format(self.event_planner.subgoal))

            # Save agent view image
            if Configuration.PRINT_CAMERA_VIEW_IMAGES:
                Logger.save_img("view_{}.png".format(i), self.event.frame)

            # Save agent depth view image
            if Configuration.PRINT_CAMERA_DEPTH_VIEW_IMAGES:
                Logger.save_img("depth_view_{}.png".format(i), (self.event.depth_frame/np.max(self.event.depth_frame)*255).astype('uint8'))

            # Observe new state
            new_state = self.observe()

            # Add state in abstract model
            self.learner.add_state(new_state)

            # Add transition
            self.learner.add_transition(self.state, event_action, new_state)

            # Update current state
            self.state = new_state

            # Update agent position in agent state and path planner state
            self.update()

            # Get new observations in form of a dataframe
            new_obs = self.GetObs.get_obs_modified(new_state)

            yield new_obs



    def step(self, action):

        action_result = None

        if action.startswith("Rotate") or action.startswith("Look"):
            if len(action.split("|")) > 1:
                degrees = round(float(action.split("|")[1]), 1)
                action_result = self.controller.step(action=action.split("|")[0], degrees=degrees)
            else:
                action_result = self.controller.step(action=action)


        elif action.startswith("PickupObject") or action.startswith("PutObject"):

            # Store held objects
            old_inventory = copy.deepcopy(self.event.metadata['inventoryObjects'])

            # If xy camera coordinates are used to perform the action, i.e., are in the action name
            if len(action.split("|")) > 2:
                x_pos = round(float(action.split("|")[1]), 2)
                y_pos = round(float(action.split("|")[1]), 2)

                # Save action outcome
                action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos, forceAction=True)

                # Hide picked up objects
                if Configuration.HIDE_PICKED_OBJECTS and action.startswith("PickupObject") \
                        and action_result.metadata['lastActionSuccess']:
                    for picked_obj in action_result.metadata['inventoryObjects']:
                        self.controller.step('HideObject', objectId=picked_obj['objectId'])

                # Unhide put down objects
                elif Configuration.HIDE_PICKED_OBJECTS and action.startswith("PutObject") \
                        and action_result.metadata['lastActionSuccess']:
                    for released_obj in old_inventory:
                        self.controller.step('UnhideObject', objectId=released_obj['objectId'])

            # Other cases
            else:
                print('You should manage the case where a pickup/putdown action is performed without '
                      'passing input xy camera coordinates. Look at step() method in Agent.py .')
                exit()

        elif action == "HomePosition":
            action_result = self.controller.step('TeleportFull', x=self.pos['x'], y=0.9009997248649597, z=self.pos['y'],
                                                 # 0.9009997248649597 is the agent starting height
                                                 rotation=dict(x=0.0, y=270.0, z=0.0),
                                                 horizon=0)
        elif action == "Stop":
            self.step("Pass")
            input('Agent has completed the plan! Press any key to exit.')
            exit()

        else:
            # Execute "move" action in the environment
            action_result = self.controller.step(action=action)

        return action_result


    def observe(self):

        # If no action has been executed yet, execute a dummy one to get current observation
        if self.event is None:
            self.event = self.controller.step("Pass")

        # Get perceptions
        x_pos = self.event.metadata['agent']['position']['x']
        y_pos = self.event.metadata['agent']['position']['z']
        camera_z_pos = self.event.metadata['cameraPosition']['y']
        hand_x_pos = self.event.metadata['hand']['position']['x']
        hand_y_pos = self.event.metadata['hand']['position']['z']
        hand_z_pos = self.event.metadata['hand']['position']['y']
        # camera_z_pos = self.event.metadata['agent']['position']['y']
        angle = self.event.metadata['agent']['rotation']['y']
        camera_angle = self.event.metadata['agent']['cameraHorizon']
        rgb_img = self.event.frame
        depth_img = self.event.depth_frame


        # Store bounding box and perceptions
        bb = self.event.instance_detections2D

        perceptions = np.concatenate((np.array([x_pos, y_pos, camera_z_pos,
                                                hand_x_pos, hand_y_pos, hand_z_pos,
                                                angle, camera_angle]), rgb_img.flatten(), depth_img.flatten()))

        #img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('img_int_{}.jpg', img)

        # Get ground truth visible object detection
        visible_objects = self.learner.object_detector.get_visible_objects_ground_truth(self.event.metadata)

        # Create new state
        s_new = State(len(self.learner.abstract_model.states), perceptions, visible_objects, bb)
        #s_new = State(len(self.learner.abstract_model.states), perceptions, self.learner.knowledge_manager.all_objects)

        return s_new


    def update_collision_map(self, agent_theta):

        Logger.write("Collision detected")

        # Map agent position into grid
        start = [self.pos['x'] * 100, self.pos['y'] * 100]
        start_grid = (int(round((start[0] - self.learner.mapper.map_model.x_min) / self.learner.mapper.map_model.dx)),
                      int(round((start[1] - self.learner.mapper.map_model.y_min) / self.learner.mapper.map_model.dy)))

        collision_cell = None

        if agent_theta == 0:
            collision_cell = [start_grid[1], start_grid[0] + 1]
        elif agent_theta == 90:
            collision_cell = [start_grid[1] + 1, start_grid[0]]
        elif agent_theta == 180:
            collision_cell = [start_grid[1], start_grid[0] - 1]
        elif agent_theta == 270:
            collision_cell = [start_grid[1] - 1, start_grid[0]]

        assert collision_cell is not None, "Cannot add null collision cell"

        self.learner.mapper.map_model.collision_cells.append(collision_cell)


    def update(self):
        """
        Update agent position in agent state and in path planner state
        :return: None
        """

        # Update agent xyz position
        self.pos['x'], self.pos['y'], self.pos['z'] = self.state.perceptions[0], self.state.perceptions[1], self.state.perceptions[2]

        # Update agent xyz position
        self.hand_pos['x'], self.hand_pos['y'], self.hand_pos['z'] = self.state.perceptions[3], self.state.perceptions[4], self.state.perceptions[5]

        # Update agent y rotation
        self.angle = (360 - int(round(self.state.perceptions[6])) + 90) % 360 # rescale angle according to standard reference system

        # Update path planner state
        self.event_planner.path_planner.agent_position = self.pos
        self.event_planner.path_planner.agent_angle = self.angle

        # Update event planner state
        self.event_planner.state = self.state


    def get_visible_objects(self):
        visible_objects = defaultdict(list)

        for obj in self.event.metadata['objects']:
            if obj['visible']:
                obj_name = "{}_{}".format(obj['objectType'].lower(), len(visible_objects[obj['objectType'].lower()]))
                obj_map_x = obj['position']['x']
                obj_map_y = obj['position']['z']
                obj_map_z = obj['position']['y']
                obj_bound_box = obj['axisAlignedBoundingBox']
                # obj_distance = obj['distance']
                visible_objects[obj['objectType'].lower()].append({"id":obj_name,
                                                           "map_x":obj_map_x,
                                                           "map_y":obj_map_y,
                                                           "map_z":obj_map_z,
                                                           "bound_box":obj_bound_box
                                                           # "distance": obj_distance}
                                                            })

        return dict(visible_objects)



    def move_objs(self):

        # Set high visibility distance to check which objects are in front of the agent
        self.controller.reset(visibilityDistance="20")
        pass_event = self.step(action="Pass")
        true_objects = pass_event.metadata["objects"]
        #true_objects = self.event.metadata["objects"]

        # Choose a random receptacle
        rec = random.choice(true_objects)
        # Check if it is receptacle and if it is not visible by the agent
        while not rec["receptacle"] or rec["visible"]:
            rec = random.choice(true_objects)
        rec_Id = rec["objectId"]
        # Get receptacle coordinates
        rec_coords = self.controller.step(action="GetSpawnCoordinatesAboveReceptacle", objectId=rec_Id, anywhere=True)
        new_pos = rec_coords.metadata["actionReturn"]

        # Choose a random object
        rand_obj = random.choice(true_objects)
        # Check if it is pickupable and if it is not visible by the agent
        while not rand_obj["pickupable"] or rand_obj["visible"]:
            rand_obj = random.choice(true_objects)
        object_Id = rand_obj["objectId"]

        # Move the object to the receptacle
        if new_pos is not None and new_pos != [] and object_Id != rec_Id:
            execute_move = self.controller.step(action="PlaceObjectAtPoint", objectId=object_Id, position=new_pos[0])
            return execute_move

        else:
            return self.step(action="Pass")



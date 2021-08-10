import math
import random
import numpy as np

from PAL.Plan.PathPlanner import PathPlanner


class EventPlanner:

    def __init__(self, map_model):

        # Set path planner
        self.path_planner = PathPlanner(map_model)

        # Set pddl plan
        self.pddl_plan = None
        # self.pddl_plan = ["GET_CLOSE_TO(APPLE_1)", "PICKUP(APPLE_1)", "GET_CLOSE_TO(BOX_1)", "PUTIN(APPLE_1, BOX_1)",
        #                   "STOP()"]

        # Set path plan
        self.path_plan = None

        # Set event plan
        self.event_plan = None

        # Current agent state
        self.state = None

        # Current event planner subgoal
        self.subgoal = None

        # Object goal position
        self.goal_obj_position = None


    def plan(self):
        action = self.explore()
        return action



    def explore(self):
        if self.path_plan is None:
            self.path_plan = self.path_planner.path_planning()

        while self.path_plan is None or len(self.path_plan)==0:
            print('Changing goal position')
            self.path_planner.goal_position = [random.randint(self.path_planner.map_model.x_min + 20,
                                                              self.path_planner.map_model.x_max - 20) / 100,
                                               random.randint(self.path_planner.map_model.y_min + 20,
                                                              self.path_planner.map_model.y_max - 20) / 100]
            print("New goal position is: {}".format(self.path_planner.goal_position))
            self.path_plan = self.path_planner.path_planning()

        return self.path_plan.pop(0)

import collections
import copy
import math

import numpy as np

from PIL import Image

import Configuration
from Utils import Logger


class PathPlanner:

    def __init__(self, map_model):
        self.map_model = map_model
        self.goal_position = [5, -2]
        self.agent_position = None
        self.agent_angle = None


    def path_planning(self):

        # Get occupancy grid
        grid = self.get_occupancy_grid()

        # Add agent starting position into occupancy grid
        start = [self.agent_position['x']*100, self.agent_position['y']*100]
        start_grid = (int(round((start[0]-self.map_model.x_min)/self.map_model.dx)),
                      int(round((start[1]-self.map_model.y_min)/self.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1]) # starting column and row of the grid

        # Add goal cell marker into occupancy grid
        goal = [self.goal_position[0]*100, self.goal_position[1]*100]
        goal_grid = [int(round((goal[0]-self.map_model.x_min)/self.map_model.dx)),
                     int(round((goal[1]-self.map_model.y_min)/self.map_model.dy))]
        grid[grid.shape[0] - goal_grid[1]][goal_grid[0]] = 2 # 2 is the goal integer identifier in the grid

        # DEBUG plot grid
        grid_debug = copy.deepcopy(grid)
        grid_debug[(grid_debug==1)] = 255
        grid_debug[(grid_debug==2)] = 180
        grid_debug[start_grid[1]][start_grid[0]] = 100
        Logger.save_img("topview_grid_noplan.png", grid_debug)

        # Compute plan into resized occupancy grid
        grid_plan = self.bfs(grid, start_grid, goal)
        plan = self.compile_plan(grid_plan)

        # Plot grid with plan for debugging
        if Configuration.PRINT_TOP_VIEW_GRID_PLAN_IMAGES:
            if grid_plan is not None:
                idx_i, idx_j = zip(*grid_plan)
                grid_debug[idx_j, idx_i] = 220 # draw plan
                grid_debug[(grid_debug==2)] = 180 # draw goal position
                grid_debug[start_grid[1]][start_grid[0]] = 100 # draw agent position
                Logger.save_img("topview_grid.png", grid_debug)

        return plan


    def path_planning_greedy(self):

        # Get occupancy grid
        grid = self.get_occupancy_grid()

        # Add agent starting position into occupancy grid
        start = [self.agent_position['x']*100, self.agent_position['y']*100]
        start_grid = (int(round((start[0]-self.map_model.x_min)/self.map_model.dx)),
                      int(round((start[1]-self.map_model.y_min)/self.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1]) # starting column and row of the grid

        # Add goal cell marker into occupancy grid
        goal = [self.goal_position[0]*100, self.goal_position[1]*100]
        goal_grid = [int(round((goal[0]-self.map_model.x_min)/self.map_model.dx)),
                     int(round((goal[1]-self.map_model.y_min)/self.map_model.dy))]
        grid[grid.shape[0] - goal_grid[1]][goal_grid[0]] = 2 # 2 is the goal integer identifier in the grid

        # Add greedy goal cells marker into occupancy grid, i.e. positions within 90 centimeters from goal point
        # and towards agent point
        for i in range(1,int(Configuration.MAX_DISTANCE_MANIPULATION/10) + 1):
        # for i in range(1,10):
            distance = 10
            goal_point_greedy = self.get_point_with_distance_from(start[0], start[1], goal[0], goal[1], distance*i)
            goal_cell_greedy = [int(round((goal_point_greedy[0]-self.map_model.x_min)/self.map_model.dx)),
                                int(round((goal_point_greedy[1]-self.map_model.y_min)/self.map_model.dy))]

            if grid[grid.shape[0] - goal_cell_greedy[1]][goal_cell_greedy[0]] != 0:
                grid[grid.shape[0] - goal_cell_greedy[1]][goal_cell_greedy[0]] = 2 # 2 is the goal integer identifier in the grid

            if grid[grid.shape[0] - goal_cell_greedy[1] +1][goal_cell_greedy[0]] != 0:
                grid[grid.shape[0] - goal_cell_greedy[1] + 1][goal_cell_greedy[0]] = 2

            if grid[grid.shape[0] - goal_cell_greedy[1] - 1][goal_cell_greedy[0]] != 0:
                grid[grid.shape[0] - goal_cell_greedy[1] - 1][goal_cell_greedy[0]] = 2

            if grid[grid.shape[0] - goal_cell_greedy[1]][goal_cell_greedy[0] + 1] != 0:
                grid[grid.shape[0] - goal_cell_greedy[1]][goal_cell_greedy[0] + 1] = 2

            if grid[grid.shape[0] - goal_cell_greedy[1]][goal_cell_greedy[0] -1] != 0:
                grid[grid.shape[0] - goal_cell_greedy[1]][goal_cell_greedy[0] -1] = 2

        # DEBUG plot grid
        grid_debug = copy.deepcopy(grid)
        grid_debug[(grid_debug==1)] = 255
        grid_debug[(grid_debug==2)] = 180
        grid_debug[start_grid[1]][start_grid[0]] = 100
        Logger.save_img("topview_grid_noplan.png", grid_debug)

        # Check if agent position is already a goal one
        if grid[start_grid[1]][start_grid[0]] == 2:
            return []

        # Compute plan into resized occupancy grid
        grid_plan = self.bfs(grid, start_grid, goal)
        plan = self.compile_plan(grid_plan)

        # Plot grid with plan for debugging
        if Configuration.PRINT_TOP_VIEW_GRID_PLAN_IMAGES:
            if grid_plan is not None:
                idx_i, idx_j = zip(*grid_plan)
                grid_debug[idx_j, idx_i] = 220 # draw plan
                grid_debug[(grid_debug==2)] = 180 # draw goal position
                grid_debug[start_grid[1]][start_grid[0]] = 100 # draw agent position
                Logger.save_img("topview_grid.png", grid_debug)

        return plan


    def get_occupancy_grid(self):
        # Rescale agent top view
        grid = Image.frombytes('RGB', self.map_model.fig.canvas.get_width_height(), self.map_model.fig.canvas.tostring_rgb()).convert('L')
        grid.thumbnail((round(self.map_model.y_axis_len/self.map_model.dy), round(self.map_model.x_axis_len/self.map_model.dx)), Image.ANTIALIAS)
        grid = np.array(grid)

        # Binarize rescaled agent top view
        grid[(grid < 235)] = 0 # 235 is an heuristic threshold
        grid[(grid >= 235)] = 1

        # Add collision cells into occupancy grid
        for cell in self.map_model.collision_cells:
            grid[grid.shape[0] - cell[0]][cell[1]] = 0

        return grid


    def bfs(self, grid, start, goal):
        """
        Example call
        wall, clear, goal = "#", ".", "*"
        width, height = 10, 5
        grid = ["..........",
                "..*#...##.",
                "..##...#*.",
                ".....###..",
                "......*..."]
        path = bfs(grid, (5, 2))
        :param grid: occupancy map
        :param start: starting grid cell
        :return:
        """
        wall = 0
        goal = 2
        height = grid.shape[0]
        width = grid.shape[1]
        queue = collections.deque([[start]])
        seen = set([tuple(start)])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if grid[y][x] == goal:
            # if [y, x] == goal:
                return path
            for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                if 0 <= x2 < width and 0 <= y2 < height \
                        and grid[y2][x2] != wall \
                        and (x2, y2) not in seen:
                        # and grid[min(y2 + 1, height - 1)][x2] != wall and grid[max(y2 - 1, 0)][x2] != wall \
                        # and grid[y2][min(x2 + 1, width - 1)] != wall and grid[y2][max(x2 - 1, 0)] != wall\
                        # and grid[min(y2 + 1, height - 1)][min(x2 + 1, width - 1)] != wall \
                        # and grid[max(y2 - 1, 0)][max(x2 - 1, 0)] != wall\
                        # and grid[min(y2 + 1, height - 1)][max(x2 - 1, 0)] != wall \
                        # and grid[max(y2 - 1, 0)][min(x2 + 1, width - 1)] != wall \
                        # and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))


    def compile_plan(self, plan):

        # If no plan can be computed, return none action list
        if plan is None:
            return None

        plan_actions = []

        dtheta = 90 # 90 degrees is the dtheta of agent rotation action
        agent_theta = copy.deepcopy(self.agent_angle)

        for i in range(len(plan) - 1):
            relative_angle = None

            # Get relative angle between two subsequent grid cells
            # plan is a list of tuples (column, row)
            if plan[i+1][1] == plan[i][1] - 1 and plan[i+1][0] == plan[i][0]:
                relative_angle = 90
            elif plan[i + 1][1] == plan[i][1] and plan[i + 1][0] == plan[i][0] - 1:
                relative_angle = 180
            elif plan[i + 1][1] == plan[i][1] + 1 and plan[i + 1][0] == plan[i][0]:
                relative_angle = 270
            elif plan[i + 1][1] == plan[i][1] and plan[i + 1][0] == plan[i][0] + 1:
                relative_angle = 0

            move_angle = relative_angle - agent_theta

            # Optimize rotations, i.e. instead of 3 rotating left actions => 1 rotating right action
            if move_angle > 180:
                move_angle = move_angle - 360
            elif move_angle < -180:
                move_angle = 360 + move_angle

            # Add rotation actions
            if move_angle > 0:
                for _ in range(abs(move_angle // dtheta)):
                    plan_actions.append('RotateLeft')
                    agent_theta += dtheta
                    agent_theta = agent_theta % 360
            else:
                for _ in range(abs(move_angle // dtheta)):
                    plan_actions.append('RotateRight')
                    agent_theta -= dtheta
                    agent_theta = agent_theta % 360

            # Add move forward action
            plan_actions.append('MoveAhead')

        return plan_actions



    def move_point_to_feasibleOLD(self, agent_x, agent_y, agent_angle):

        # Get occupancy grid
        occupancy_grid = self.get_occupancy_grid()

        # Get goal cell in occupancy grid
        agent_x, agent_y = agent_x * 100, agent_y * 100
        goal_point = [self.goal_position[0] * 100, self.goal_position[1] * 100]
        goal_cell = [int(round((goal_point[0]-self.map_model.x_min)/self.map_model.dx)),
                     int(round((goal_point[1]-self.map_model.y_min)/self.map_model.dy))]

        # Get angle between agent position and goal position
        obj_agent_angle = np.rad2deg(math.atan2(goal_point[1] - agent_y, goal_point[0] - agent_x))
        # Add agent angle offset
        obj_agent_angle = obj_agent_angle - agent_angle

        distance_step = 10 # 10 centimeters
        while occupancy_grid[occupancy_grid.shape[0] - goal_cell[1]][goal_cell[0]] == 0:

            # goal_point[0] = goal_point[0] + distance_step*np.cos(np.deg2rad(obj_agent_angle))
            # goal_point[1] = goal_point[1] + distance_step*np.sin(np.deg2rad(obj_agent_angle))

            goal_point[0] = goal_point[0] - distance_step*np.cos(np.deg2rad(obj_agent_angle))
            goal_point[1] = goal_point[1] - distance_step*np.sin(np.deg2rad(obj_agent_angle))

            goal_cell = [int(round((goal_point[0]-self.map_model.x_min)/self.map_model.dx)),
                         int(round((goal_point[1]-self.map_model.y_min)/self.map_model.dy))]

        return [goal_point[0]/100, goal_point[1]/100]



    def move_point_to_feasible(self, agent_x, agent_y, agent_angle):

        # Get occupancy grid
        occupancy_grid = self.get_occupancy_grid()

        # Get goal cell in occupancy grid
        agent_x, agent_y = agent_x * 100, agent_y * 100
        goal_point = [self.goal_position[0] * 100, self.goal_position[1] * 100]
        goal_cell = [int(round((goal_point[0]-self.map_model.x_min)/self.map_model.dx)),
                     int(round((goal_point[1]-self.map_model.y_min)/self.map_model.dy))]

        distance_step = 10 # 10 centimeters
        x1, y1 = agent_x, agent_y
        while occupancy_grid[occupancy_grid.shape[0] - goal_cell[1]][goal_cell[0]] == 0:
            goal_point = self.get_point_with_distance_from(agent_x, agent_y, goal_point[0], goal_point[1], distance_step)
            # x0, y0 = goal_point[0], goal_point[1]
            # v = [x1 - x0, y1 - y0]
            # # u = v/norm(v)
            # u = [v[0] / np.linalg.norm(np.array(v)), v[1] / np.linalg.norm(np.array(v))]
            # goal_point[0] = goal_point[0] + distance_step*u[0]
            # goal_point[1] = goal_point[1] + distance_step*u[1]

            goal_cell = [int(round((goal_point[0]-self.map_model.x_min)/self.map_model.dx)),
                         int(round((goal_point[1]-self.map_model.y_min)/self.map_model.dy))]

        return [goal_point[0]/100, goal_point[1]/100]


    def get_point_with_distance_from(self, start_x, start_y, end_x, end_y, distance):

        goal_point = [end_x, end_y]
        x0, y0 = end_x, end_y
        x1, y1 = start_x, start_y
        v = [x1 - x0, y1 - y0]
        # u = v/norm(v)
        u = [v[0] / np.linalg.norm(np.array(v)), v[1] / np.linalg.norm(np.array(v))]

        # Move goal point towards start point
        goal_point[0] = goal_point[0] + distance*u[0]
        goal_point[1] = goal_point[1] + distance*u[1]

        return goal_point




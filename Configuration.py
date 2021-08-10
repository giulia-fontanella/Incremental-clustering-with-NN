# IP address of WSL2 in Windows
IP_ADDRESS = "172.29.48.1" # Set this for WSL2 by looking : cat /etc/resolv.conf


##########################################################
################### RUN CONFIGURATION ####################
##########################################################
MAX_ITER = 500


##########################################################
############## ITHOR SIMULATOR CONFIGURATION #############
##########################################################
RENDER_DEPTH_IMG = 1
HIDE_PICKED_OBJECTS = 1


##########################################################
################## LOGGER CONFIGURATION ##################
##########################################################

# Print output information
VERBOSE = 1

# Save images
PRINT_IMAGES = 1

# Save agent camera view images
PRINT_CAMERA_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save agent camera depth view images
PRINT_CAMERA_DEPTH_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save top view images
PRINT_TOP_VIEW_IMAGES = 1 and PRINT_IMAGES

# Save top view images
PRINT_TOP_VIEW_GRID_PLAN_IMAGES = 1 and PRINT_IMAGES


##########################################################
################ MAP MODEL CONFIGURATION #################
##########################################################

# x min coordinate in centimeters
MAP_X_MIN = -500

# y min coordinate in centimeters
MAP_Y_MIN = -1000

# x max coordinate in centimeters
MAP_X_MAX = 1500

# y max coordinate in centimeters
MAP_Y_MAX = 1000

# x and y centimeters per pixel in the resized grid occupancy map
MAP_GRID_DX = 10
MAP_GRID_DY = 10


##########################################################
############# OBJECT DETECTOR CONFIGURATION ##############
##########################################################
MIT_SEM_SEG = False
FASTER_RCNN = True
OBJ_SCORE_THRSH = 0.3

##########################################################
############### PATH PLANNER CONFIGURATION ###############
##########################################################
MAX_DISTANCE_MANIPULATION = 110 # centimeters


##########################################################
############### PDDL PLANNER CONFIGURATION ###############
##########################################################
FF_PLANNER = "FF"
FD_PLANNER = "FD"
PLANNER_TIMELIMIT = 300
PLANNER = FF_PLANNER
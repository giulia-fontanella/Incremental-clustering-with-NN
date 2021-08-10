import os
import cv2
import Configuration
from PIL import Image

# Create results root directory
os.makedirs("Results", exist_ok=True)
LOG_DIR_PATH = os.path.join("Results", "episode_{}".format(len(os.listdir("Results"))))
# Create episode results directory
os.mkdir(LOG_DIR_PATH)
LOG_FILE = open(os.path.join(LOG_DIR_PATH, "log.txt"), "w")


def save_img_cv2(img_name, cv2img):
    # Save image in episode results directory
    if Configuration.PRINT_IMAGES:
        cv2.imwrite(os.path.join(LOG_DIR_PATH, img_name), cv2img)


def save_img(img_name, img_array):
    # Save image in episode results directory
    if Configuration.PRINT_IMAGES:
        im = Image.fromarray(img_array)

        # Save grayscale image
        if len(img_array.shape) < 3:
            im = im.convert("L")

        im.save(os.path.join(LOG_DIR_PATH, img_name))


def write(string):
    # Print string in log file
    LOG_FILE.write("\n" + string)

    # Print string in console
    if Configuration.VERBOSE:
        print(string)
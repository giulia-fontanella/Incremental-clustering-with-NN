from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Function to extract features from an rgb image, using a pretrained network (vgg16)

def extract_features(img):
    # Initialize the model
    model = VGG16(weights='imagenet', include_top=False)#, Pooling=None)
    #model.summary()

    # Add an axis
    x = np.expand_dims(img, axis=0)
    # Preprocess input
    x = preprocess_input(x)
    feat = model(x)
    return feat


# Prova di estrazione features
#import pickle
#with open("../dataset/clustering_FloorPlan17.pkl", "rb") as input_file:
#    dataset2 = pickle.load(input_file)
#id = list(dataset2.keys())[0]
#obs_list = dataset2[id]
#img = obs_list[0][3]
#feat = extract_features(img)
#print(np.shape(feat))
#print(feat)


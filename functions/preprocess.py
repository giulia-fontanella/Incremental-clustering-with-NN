import numpy as np
import pandas as pd
import random

# Function to transform the output of agent.run to a dataframe ready for neural network
# In input we have a dataframe with observations on each row
# In output we have a dataframe with couples of observations on the same row, plus binary indicator if they are
# in the same cluster


def preprocess(dataframe, max_selected_obs, num_ex):

    rows = np.shape(dataframe)[0]
    cols = np.shape(dataframe)[1]
    all_obs = []

    # Iterate over the observations and form couples, without repetitions

    # FIRST METHOD
    # Select only "max_couples" observations to pair with other "max_couple" observations
    #for i in random.sample(range(rows), min(int(max_couples/2), rows)):
    #    for j in random.sample(range(i, rows), min(int(max_couples/2), rows-i)):

    # SECOND METHOD
    # Select some positive examples to make the dataset balanced
    for i in random.sample(range(rows), min(max_selected_obs, rows)):

        # Select num_ex ajacent rows, which will probably be in the same cluster, and num_ex random rows
        indices = list(range(i-num_ex//2, i+num_ex//2)) + random.sample(range(rows), num_ex-4)
        for j in indices:
        # Check if j is a row
            if j in range(rows):

                # Define a binary variable y which indicates if two obs are in the same cluster or not
                id1 = dataframe.iloc[i, cols - 1]
                id2 = dataframe.iloc[j, cols - 1]
                y = pd.Series([int(id1 == id2)])

                # Select the 2 observations from the dataframe
                obs1 = dataframe.iloc[i, 0 : cols-2]
                obs2 = dataframe.iloc[j, 0 : cols-2]

                # Put them together in a new row with binary variable
                new_row = pd.concat([obs1, obs2, y], ignore_index=True)
                del id1, id2, y, obs1, obs2
                all_obs.append(new_row)


    # Create a new dataframe: each row has 2 observations (3 obj positions, 3 agent positions, 2 angles, 512 features)
    # + y binary value -> 520 * 2 + 1 = 1041
    # Shape (n, 1041)
    preprocessed_dataframe = pd.DataFrame(all_obs)
    return preprocessed_dataframe



# Function to create new couples of observations in the second part of the algorithm.
# It takes in input a dataframe with previous observations + the dataframe collected in this iteration.

def preprocess_modified(dataframe1, dataframe2, max_couples):

    rows1 = np.shape(dataframe1)[0]
    rows2 = np.shape(dataframe2)[0]
    cols = np.shape(dataframe1)[1]
    all_obs = []

    # Iterate over the observations and form couples, without repetitions
    # Select only "max_couples" observations to pair with other "max_couple" observations
    for i in random.sample(range(rows1), min(max_couples, rows1)):
        for j in random.sample(range(rows2), min(max_couples, rows2)):
            # Define a binary variable y which indicates if two obs are in the same cluster or not
            id1 = dataframe1.iloc[i, cols - 1]
            id2 = dataframe2.iloc[j, cols - 1]
            y = pd.Series([int(id1 == id2)])

            # Select the 2 observations from the dataframe
            obs1 = dataframe1.iloc[i, 0: cols - 1]
            obs2 = dataframe2.iloc[j, 0: cols - 1]

            # Put them together in a new row with binary variable
            new_row = pd.concat([obs1, obs2, y], ignore_index=True)
            del id1, id2, y, obs1, obs2
            all_obs.append(new_row)


    # Create a new dataframe: each row has 2 observations (3 positions+512 features) + y binary value.

    preprocessed_dataframe = pd.DataFrame(all_obs)
    del all_obs

    return preprocessed_dataframe



# Function for testing phase

def preprocess_test(observation, dataframe):

    rows = np.shape(dataframe)[0]
    cols = np.shape(dataframe)[1]
    all_obs = []

    # Iterate over the observations and form couples
    for i in range(rows):
        # Define a binary variable y which indicates if two obs are in the same cluster or not
        id1 = observation[-1]
        id2 = dataframe.iloc[i, cols - 1]
        y = pd.Series([int(id1 == id2)])

        # Select the 2 observations from the dataframe
        obs1 = pd.Series(observation[0: cols - 1])
        obs2 = dataframe.iloc[i, 0: cols - 1]

        # Put them together in a new row with binary variable
        new_row = pd.concat([obs1, obs2, y], ignore_index=True)
        del id1, id2, y, obs1, obs2
        all_obs.append(new_row)

    # Create the new dataframe
    preprocessed_dataframe = pd.DataFrame(all_obs)
    del all_obs

    return preprocessed_dataframe


# Function to pair all observation of a dataframe

def preprocess_all(dataframe):

    rows = np.shape(dataframe)[0]
    cols = np.shape(dataframe)[1]
    all_obs = []

    # Iterate over all the observations and form couples
    for i in range(rows):
        for j in range(rows):
            if i != j:
                # Define a binary variable y which indicates if two obs are in the same cluster or not
                id1 = dataframe.iloc[i, cols - 1]
                id2 = dataframe.iloc[j, cols - 1]
                y = pd.Series([int(id1 == id2)])

                # Select the 2 observations from the dataframe
                obs1 = dataframe.iloc[i, 0: cols - 1]
                obs2 = dataframe.iloc[j, 0: cols - 1]

                # Put them together in a new row with binary variable
                new_row = pd.concat([obs1, obs2, y], ignore_index=True)
                del id1, id2, y, obs1, obs2
                all_obs.append(new_row)

    # Create the new dataframe
    preprocessed_dataframe = pd.DataFrame(all_obs)
    del all_obs

    return preprocessed_dataframe
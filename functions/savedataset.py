import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Function to save the output of Agent.run to files


def savedataset(result0, result1, scene, plot_obj_types):

    # Open files
    outfile1 = h5py.File("dataset/dataset_{}.hdf5".format(scene), "w")
    outfile2 = open("dataset/groundtruth_{}.pkl".format(scene), "wb")
    outfile3 = open("dataset/bbs_{}.pkl".format(scene), "wb")

    # Save results in lists
    perceptions = []
    visible_objects = []
    bbs = []
    for state in result0:
        perceptions.append(state.perceptions)
        visible_objects.append(state.visible_objects)
        bbs.append(state.bb)

    newperc = np.vstack(perceptions)

    # Save perceptions in h5py
    dset = outfile1.create_dataset("Perceptions", data=newperc, compression="gzip")
    # To read: with h5py.File('dataset/dataset_FloorPlan17.hdf5', 'r') as f:
    #    data = f['Perceptions'][:]

    # Save visible objects and bbs in pickle
    pickle.dump(visible_objects, outfile2)
    pickle.dump(bbs, outfile3)

    # Close files
    outfile1.close()
    outfile2.close()
    outfile3.close()

    # Clear memory
    del perceptions, visible_objects, bbs, newperc

    # Save initial clustering in a list
    initial_clustering = []
    clustering = result1
    obj_count = {}
    for cluster in clustering.clusters:
        #print(cluster.name, len(cluster.observations))
        for obs in cluster.observations:
            new_row = obs.tolist()
            new_row.append(cluster.name)
            new_row.append(cluster.ID)
            initial_clustering.append(new_row)

        # Select object type, number of observation and add it to a dict, to keep count of distribution of
        # observations across different object types
        obj_type = cluster.name.partition("|")[0]
        length = len(cluster.observations)
        if obj_type not in obj_count.keys():
            obj_count[obj_type] = length
        else:
            obj_count[obj_type] += length

    # Plot an histogram of object types observations
    plt.figure(figsize=(7, 6))
    mpl.rcParams['xtick.labelsize'] = 7
    values = pd.Series(obj_count.values())
    labels = list(obj_count.keys())
    ax = values.plot(kind='bar')
    ax.set_xticklabels(labels)
    if plot_obj_types==True:
        plt.savefig('plots/objtypes/Object_types_{}.jpg'.format(scene))
    plt.close('all')

    # Convert to dataframe and save
    dataframe = pd.DataFrame(initial_clustering)
    dataframe.to_pickle("dataset/clustering_{}.pkl".format(scene))

    # Clear memory
    del initial_clustering, clustering, dataframe


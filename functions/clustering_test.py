import numpy as np
import scipy.spatial.distance as distance


class Clustering:

    def __init__(self):
        # Initialize clusters as a dictionary
        self.clusters = {}
        self.clustering_list = []


    def initialize(self, index, observation):
            self.clusters[index] = [observation]


    def add_observation(self, obs, cluster_id):
        # Append observation to cluster corresponding to key "cluster_id"
        if cluster_id in self.clusters.keys():
            self.clusters[cluster_id].append(obs)
        else:
            self.clusters[cluster_id] = [obs]



    def update_clusters(self, max_len):
        # Remove observations to stay under a max number of elements per cluster
        for cluster in self.clusters.values():          # cluster is the list of observations
            diff = len(cluster) - max_len
            # If len cluster is greater than max_len, calculate distances and remove closer observations
            if diff > 0:
                for i in range(diff):
                    X = np.array(cluster)
                    # Calculate pairwise cosine distance and obtain distance matrix
                    dist_array = distance.pdist(X[:, 3:-1], metric="cosine")
                    dist_matrix = distance.squareform(dist_array)
                    # Add diagonal which otherwise is 0
                    dim = np.shape(dist_matrix)[0]
                    dist_matrix += 100*np.eye(dim)
                    # Identify minimum element row index (take the first one with [0][0], it's the same)
                    index = np.where(dist_matrix == np.amin(dist_matrix))[0][0]
                    # Find element in the list and remove
                    del cluster[index]


    def update_list(self):
        # Update clustering_list. Convert dict to a list of all observations
        self.clustering_list = []
        for cluster_id, value in self.clusters.items():
            #new_val = value + [cluster_id]
            self.clustering_list.append(value)



class Cluster:

    def __init__(self, name, observations, ID):
        self.name = name                    # object id
        self.ID = ID                        # id for the cluster
        self.observations = observations    # list of observations

    def add_observation(self, obs):
        self.observations.append(obs)

from PAL.Learn.EnvironmentModels.AbstractModel import AbstractModel
from PAL.Learn.KnowledgeManager import KnowledgeManager
from PAL.Learn.Mapper import Mapper
from PAL.Learn.ObjectDetector import ObjectDetector


class Learner:

    def __init__(self):

        # Abstract model (Finite state machine)
        self.abstract_model = AbstractModel()

        # Depth mapper
        self.mapper = Mapper()

        # Object detector
        self.object_detector = ObjectDetector()

        # Knowledge manager
        self.knowledge_manager = KnowledgeManager()


    def add_state(self, state_new):
        self.abstract_model.add_state(state_new)


    def add_transition(self, state_src, action, state_dest):
        self.abstract_model.add_transition(state_src, action, state_dest)


    def update_topview(self, file_name, depth_matrix, angle, pos):
        self.mapper.update_topview(depth_matrix, file_name, angle, pos) # depth_matrix in meters









# Incremental clustering of the perceptions of an agent with neural networks


This algorithm was developed as a framework for incremental clustering of perceptions of an agent. 
An agent is moving around an apartment collecting data about the ob-jects it sees in the rooms. The aim is to create a clustering which puts together all the different perceptions of an object. It is assumed that the objects can be moved around the room bysome other entity. The agent should recognize that an object is the same it has previously seen, even if there are changes in the position of the object or in the angulation from which it is observed. 
The algorithm which works incrementally, it can continue to learn at every step, adding more knowledge as the agent proceed in the exploration of a room. 
This algorithm will operate online, as opposed to classic offline clustering algorithms. 
The algorithm will be deployed to explore a new unseen room, building clusters of perceptions as the agent moves around.

The file 'train.py' performs the pre-training of a neural network to recognize if two perceptions are relative to a same object or not.

The file 'test_online.py' performs the clustering online, during the agent exploration of the new rooms.





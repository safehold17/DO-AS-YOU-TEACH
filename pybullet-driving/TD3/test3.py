import simple_driving
# from env import DroneEnv
from alice import *
from bob import *
import numpy as np
from copy import deepcopy
import gym
import time

env = gym.make('SimpleDriving-v0')
action_scale = np.array([0.5,0.6])
action_add = np.array([1,0])

bob = BobAgent(action_scale, action_add)

trials = 25

for t in range(4000,-1,-500):
    success = 0
    np.random.seed(1)
    modelpath = "./checkpoints/bob"+str(t)
    bob.TD.load(modelpath)
    for i in range(trials):
        # spawn_position = np.array([1,2,0.5])
        spawn_position = np.random.uniform(-10,10, (3,))
        spawn_orientation = np.random.uniform(-1,1, (4,))
        # spawn_orientation = np.zeros((4,))
        spawn_position[2] = 0.5
        goal = np.random.uniform(-10,10, (2,))
        gridmap, state = env.reset(goal,spawn_position,spawn_orientation, agent = "alice")
        gridmap, state = env.reset(goal,spawn_position,spawn_orientation, agent = "bob")
        done = False
        while not done:
            action = bob.TD.select_action(np.hstack((state, goal)),gridmap)
            # action = action * action_scale + action_add
            next_gridmap, next_state, reward, done,_ = env.step(action, 1.5, agent = "bob")
            reward = reward if (reward == -3 or reward>0) else 0
            state = deepcopy(next_state)
            gridmap = deepcopy(next_gridmap)
        if reward > 0:
            success += 1
    print(t,"------>",100*success/trials)

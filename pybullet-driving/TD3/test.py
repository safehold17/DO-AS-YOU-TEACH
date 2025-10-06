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
alice = AliceAgent(action_scale, action_add)

trials = 100

np.random.seed(5)

modelpath = "./checkpoints/bob6500"
modelpath_alice = "./checkpoints/alice6500"
bob.TD.load(modelpath)
alice.TD.load(modelpath_alice)

for i in range(trials):
    # spawn = np.array([0,0,-2])
    spawn_position = np.random.uniform(-10,10, (3,))
    spawn_orientation = np.random.uniform(-1,1, (4,))
    # spawn_position = np.array([-3,4,0.7])
    spawn_position[2] = 0.5
    gridmap, state = env.reset(12*np.ones((2,)),spawn_position, spawn_orientation)
    reward = 0
    state = np.hstack((state, spawn_position[:2]))
    while reward>=0:
        # tic = time.time()
        action = alice.TD.select_action(state, gridmap)
        # print(action)
        next_gridmap, next_state, reward, done,_ = env.step(action)
        next_state = np.hstack((next_state, spawn_position[:2]))
        state = deepcopy(next_state)
        gridmap = deepcopy(next_gridmap)
    # time.sleep(0.5)
    #run bob if alice sets valid goal
    if reward != -3:
        goal = deepcopy(state[:2])
        gridmap, state = env.reset(goal, spawn_position,spawn_orientation, agent = "bob")
        done = False
        # print("bob")
        while not done:
            action = bob.TD.select_action(np.hstack((state, goal)),gridmap)
            # print(action)
            next_gridmap, next_state, reward, done,_ = env.step(action, 1.5, agent = "bob")
            reward = reward if (reward == -3 or reward>0) else 0
            state = deepcopy(next_state)
            gridmap = deepcopy(next_gridmap)
            # time.sleep(0.01)
            
        # print(goal, state, action)
        # if done:
        #     print(reward)
    time.sleep(0.5)

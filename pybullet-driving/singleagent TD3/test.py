import simple_driving
# from env import DroneEnv
from alice import *
from bob import *
import numpy as np
from copy import deepcopy
import gym
import time

env = gym.make('SimpleDriving-v0')
bob = BobAgent()
alice = AliceAgent()

trials = 10

np.random.seed(2)

# modelpath = "./checkpoints/bob9000"
# modelpath_alice = "./checkpoints/alice9000"
# bob.TD.load(modelpath)
# alice.TD.load(modelpath_alice)
action_scale = np.array([1,0.6])
action_add = np.array([0,0])
for i in range(trials):
    # spawn = np.array([0,0,-2])
    spawn_position = np.random.uniform(-7,7, (3,))
    spawn_position[2] = 0.1
    
    goal = np.random.uniform(-10,10, (2,))
    # goal[2] = -5*np.abs(goal[2])
    
    state = env.reset(goal,spawn_position)
    # state = Denv.reset(spawn, goal)
    done = False
    # spawn = np.random.uniform(-50,50, (3,))
    # spawn[2] = -np.abs(spawn[2])
    # state = Denv.reset(spawn)
    # reward = 0
    # while reward>=0:
    #     # print(state)
    #     action = alice.TD.select_action(state).astype(float)
    #     next_state, reward, done = Denv.step(action)
    #     alice.memorize(state, action, 0, next_state)
    #     state = deepcopy(next_state)
    #     print(state, action)

    # goal = deepcopy(state)
    # done = False
    # state = Denv.reset(spawn, goal)


    while not done:
        # print(state)
        action = bob.TD.select_action(np.hstack((state, goal)))
        # action = alice.TD.select_action(state)
        action = action * action_scale + action_add
        # print(action)
        next_state, reward, done,_ = env.step(action.astype(float), 2, agent = "bob")
        state = deepcopy(next_state)
        time.sleep(0.01)
        # print(goal, state, action)
        # if done:
        #     print(reward)
    time.sleep(0.5)

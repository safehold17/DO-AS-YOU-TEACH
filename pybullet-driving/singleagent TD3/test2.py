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
bob = BobAgent(action_scale,action_add)
# alice = AliceAgent()

trials = 300
success = 0
np.random.seed(2)

modelpath = "./checkpoints/bob60000"
# modelpath_alice = "./checkpoints/alice52500"
bob.TD.load(modelpath)
# alice.TD.load(modelpath_alice)
for i in range(trials):
    bob_error = [0,0]
    spawn_position = np.random.uniform(-10,10, (3,))
    spawn_orientation = np.random.uniform(-1,1, (4,))
    # spawn_position = np.array([-3,4,0.7])
    spawn_position[2] = 0.5
    goal = np.random.uniform(-10,10, (2,))
    gridmap, state = env.reset(goal, spawn_position,spawn_orientation, agent = "bob")
    done = False
    # print("bob")
    while not done:
        # print(goal, state)
        # print(gridmap.shape)
        action = bob.TD.select_action(np.hstack((state, goal)),gridmap)
        # action_ = action * action_scale + action_add
        next_gridmap, next_state, reward, done,_ = env.step(action, 1.5, agent = "bob")
        reward = reward if (reward == -3 or reward>0) else 0
        # bob.memorize(gridmap, state, *action, reward, next_gridmap, next_state, goal, spawn_position[:2], not done)
        state = deepcopy(next_state)
        gridmap = deepcopy(next_gridmap)
    success = success+1 if reward==5 else success
    if i%10 == 9:
        print(success, "in", i+1, "-->", success*100/(i+1))
    # time.sleep(0.5)

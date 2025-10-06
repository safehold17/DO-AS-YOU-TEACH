import simple_driving
from alice import *
from bob import *
import numpy as np
from copy import deepcopy
import time
import gym
import matplotlib.pyplot as plt
# time.sleep(30)

EPISODES = 60001  # number of episodes
SIGMA = 0.2
MAX_OLD_ALICE_LENGTH = 100

action_scale = np.array([0.5,0.6])
action_add = np.array([1,0])

env = gym.make('SimpleDriving-v0')
bob = BobAgent(action_scale, action_add)

# checkpoint_path_bob = "./checkpoints/bob6000"
# checkpoint_path_alice = "./checkpoints/alice6000"

# bob.TD.load(checkpoint_path_bob)
# alice.TD.load(checkpoint_path_alice)
# logfile = open("logfile.txt", 'w')
bob_actor_err = []
bob_critic_err = []
bob_rewards = []
fig = plt.figure()


for episode in range(EPISODES):
    # alice_reward = 0
    bob_reward = 0
    # alice_error = [0,0]
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
        action = bob.TD.select_action(np.hstack((state, goal)),gridmap) + np.random.normal(0, SIGMA, (1,2))
        # action_ = action * action_scale + action_add
        next_gridmap, next_state, reward, done,_ = env.step(*action, 1.5, agent = "bob")
        # plt.imshow(next_gridmap); plt.show()
        # plt.imshow(next_gridmap==0); plt.show()
        # plt.imshow(next_gridmap==2); plt.show()
        # plt.imshow(next_gridmap==3); plt.show()
        # reward = reward if (reward == -3 or reward>0) else 0
        bob.memorize(gridmap, state, *action, reward, next_gridmap, next_state, goal, spawn_position[:2], not done)
        bob_reward = bob_reward + reward
        state = deepcopy(next_state)
        gridmap = deepcopy(next_gridmap)
        # time.sleep(0.01)
    
    # print(reward)
    # if reward>0:
    #     bob.memory[-1][3] = 5
    #     bob_reward = bob_reward+5
    # print(bob.memory)
    # print(alice.memory)
    # alice_actor_error_ , alice_critic_error_ = alice.learn()
    bob_actor_error_ , bob_critic_error_ = bob.learn()

    # alice_error[0] += alice_actor_error_
    # alice_error[1] += alice_critic_error_
    bob_error[0] += bob_actor_error_
    bob_error[1] += bob_critic_error_

    if episode%100 == 0:
        bob_actor_err.append(bob_error[0])
        bob_critic_err.append(bob_error[1])

        plt.plot(bob_actor_err, label = "bob actor")
        plt.plot(bob_critic_err, label = "bob critic")
        plt.legend()
        plt.savefig("./errorplot.png")
        fig.clf()
        
        bob_rewards.append(bob_reward)
        plt.plot(bob_rewards)
        plt.savefig("./rewardplot.png")
        fig.clf()

    print("rewards episode", episode, "bob:", bob_reward)
    if episode % 500 == 0:
        # alice.save_model('checkpoints/alice'+str(episode)+".pt")
        # bob.save_model('checkpoints/bob'+str(episode)+".pt")
        # alice.TD.save('checkpoints/alice'+str(episode))
        bob.TD.save('checkpoints/bob'+str(episode))
    

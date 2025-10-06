import simple_driving
from alice import *
from bob import *
import numpy as np
from copy import deepcopy
import time
import gym
import matplotlib.pyplot as plt
# time.sleep(30)

EPISODES = 30001  # number of episodes
SIGMA = 0.2
MAX_OLD_ALICE_LENGTH = 100

BOB_REWARD_FOR_REACHING = 5
ALICE_REWARD_FOR_REACHING =-3
ALICE_REWARD_FOR_NOT_REACHING =5
TIMESTEP_REWARD = -0.001

action_scale = np.array([0.5,0.6])
action_add = np.array([1,0])

env = gym.make('SimpleDriving-v0')
alice = AliceAgent(action_scale, action_add)
bob = BobAgent(action_scale, action_add)

# checkpoint_path_bob = "./checkpoints/bob6000"
# checkpoint_path_alice = "./checkpoints/alice6000"

# bob.TD.load(checkpoint_path_bob)
# alice.TD.load(checkpoint_path_alice)
# logfile = open("logfile.txt", 'w')
old_alice = [deepcopy(alice)]
alice_actor_err = []
bob_actor_err = []
alice_critic_err = []
bob_critic_err = []
fig = plt.figure()

for episode in range(EPISODES):
    alice_reward = 0
    bob_reward = 0
    alice_error = [0,0]
    bob_error = [0,0]
    for i in range(4):
        # run Alice
        spawn_position = np.random.uniform(-10,10, (3,))
        spawn_orientation = np.random.uniform(-1,1, (4,))
        # spawn_position = np.array([-3,4,0.7])
        spawn_position[2] = 0.5
        gridmap, state = env.reset(12*np.ones((2,)),spawn_position, spawn_orientation)
        reward = 0
        # state = np.hstack((state, spawn_position[:2]))
        while reward>=0:
            action = alice.TD.select_action(state, gridmap) + np.random.normal(0, SIGMA, (1,2))
            next_gridmap, next_state, reward, done,_ = env.step(*action)
            alice.memorize(gridmap, state, *action, 0, next_gridmap, next_state, not done)
            state = deepcopy(next_state)
            gridmap = deepcopy(next_gridmap)
            
        #run bob if alice sets valid goal
        if reward != -3:
            goal = deepcopy(state[:2])
            gridmap, state = env.reset(goal, spawn_position,spawn_orientation, agent = "bob")
            done = False
            while not done:
                action = bob.TD.select_action(np.hstack((state, goal)),gridmap) + np.random.normal(0, SIGMA, (1,2))
                next_gridmap, next_state, reward, done,_ = env.step(*action, 1.5, agent = "bob")
                reward = reward if (reward == -3 or reward>0) else TIMESTEP_REWARD
                bob.memorize(gridmap, state, *action, reward, next_gridmap, next_state, goal, not done)
                state = deepcopy(next_state)
                gridmap = deepcopy(next_gridmap)
            
            if reward>0:
                bob.memory[-1][3] = BOB_REWARD_FOR_REACHING
                bob_reward = bob_reward+BOB_REWARD_FOR_REACHING
                alice.memory[-1][3] = ALICE_REWARD_FOR_REACHING
                alice_reward = alice_reward + ALICE_REWARD_FOR_REACHING 
            else:
                alice.memory[-1][3] = ALICE_REWARD_FOR_NOT_REACHING
                alice_reward = alice_reward+ALICE_REWARD_FOR_NOT_REACHING
        # print(bob.memory)
        # print(alice.memory)
        alice_actor_error_ , alice_critic_error_ = alice.learn()
        bob_actor_error_ , bob_critic_error_ = bob.learn(alice)

        alice_error[0] += alice_actor_error_
        alice_error[1] += alice_critic_error_
        bob_error[0] += bob_actor_error_
        bob_error[1] += bob_critic_error_

    if episode%100 == 0:
        alice_actor_err.append(alice_error[0])
        alice_critic_err.append(alice_error[1])
        bob_actor_err.append(bob_error[0])
        bob_critic_err.append(bob_error[1])

        plt.plot(alice_actor_err, label = "alice actor")
        plt.plot(bob_actor_err, label = "bob actor")
        plt.plot(alice_critic_err, label = "alice critic")
        plt.plot(bob_critic_err, label = "bob critic")
        plt.legend()
        plt.savefig("./errorplot.png")
        fig.clf()
        # print("Episode time:", time.time() - tic)
### RUN OLD ALICE


    if episode%500 == 0:
        alice_ = deepcopy(alice)
        alice_.memory = []    
        if len(old_alice)<MAX_OLD_ALICE_LENGTH:
            old_alice.append(deepcopy(alice_))
        else:
            r = np.random.uniform(0,1)
            if r<0.3:
                old_alice[np.random.randint(0, MAX_OLD_ALICE_LENGTH)] = deepcopy(alice_)
        for j in range(4):
            alice_old = old_alice[np.random.randint(0, len(old_alice))]
            for i in range(2):
                #print("run")
                # run Alice
                spawn_position = np.random.uniform(-10,10, (3,))
                spawn_orientation = np.random.uniform(-1,1, (4,))
                # spawn_position = np.array([-3,4,0.7])
                spawn_position[2] = 0.5
                gridmap, state = env.reset(15*np.ones((2,)),spawn_position, spawn_orientation)
                reward = 0
                # state = np.hstack((state, spawn_position[:2]))
                while reward>=0:
                    action = alice_old.TD.select_action(state,gridmap) + np.random.normal(0, SIGMA, (1,2))
                    # action_ = action * action_scale + action_add
                    next_gridmap ,next_state, reward, done,_ = env.step(*action)
                    # next_state = np.hstack((next_state, spawn_position[:2]))
                    alice.memorize(gridmap, state, *action, 0, next_gridmap, next_state, not done)
                    state = deepcopy(next_state)
                    gridmap = deepcopy(next_gridmap)

                if reward != -3:
                    goal = deepcopy(state[:2])
                    gridmap, state = env.reset(goal, spawn_position, spawn_orientation, agent = "bob")
                    done = False
                    while not done:
                        # print(goal, state)
                        action = bob.TD.select_action(np.hstack((state, goal)), gridmap) + np.random.normal(0, SIGMA, (1,2))
                        # action_ = action * action_scale + action_add
                        next_gridmap, next_state, reward, done,_ = env.step(*action, 1.5, agent = "bob")
                        reward = reward if (reward == -3 or reward>0) else TIMESTEP_REWARD
                        bob.memorize(gridmap, state, *action, reward, next_gridmap, next_state, goal, not done)
                        state = deepcopy(next_state)
                        gridmap = deepcopy(next_gridmap)
                        # time.sleep(0.01)

                    if reward>0:
                        bob.memory[-1][3] = BOB_REWARD_FOR_REACHING
                        alice.memory[-1][3] = ALICE_REWARD_FOR_REACHING
                    else:
                        alice.memory[-1][3] = ALICE_REWARD_FOR_NOT_REACHING
                
            bob.learn(alice)
            alice.learn()
            


        # print("episode", episode,"part:", i, "-- Alice error:", alice_error, "-- Bob error:", bob_error)
    # logfile.write("rewards episode " + str(episode) + " alice: " + str(alice_reward) + " bob: "+ str(bob_reward) + "\n")
    print("rewards episode", episode, "alice:", alice_reward, "bob:", bob_reward)
    if episode % 500 == 0:
        # alice.save_model('checkpoints/alice'+str(episode)+".pt")
        # bob.save_model('checkpoints/bob'+str(episode)+".pt")
        alice.TD.save('checkpoints/alice'+str(episode))
        bob.TD.save('checkpoints/bob'+str(episode))
    

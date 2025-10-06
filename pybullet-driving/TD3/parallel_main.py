import simple_driving
from alice import *
from bob import *
import numpy as np
from copy import deepcopy
import time
import gym
import multiprocessing as mp

from run_agents import run_agents, f

# time.sleep(30)

EPISODES = 30001  # number of episodes
SIGMA = 0.2
MAX_OLD_ALICE_LENGTH = 100
pool =  mp.Pool(processes = 4)

env = gym.make('SimpleDriving-v0')
envs = []
for i in range(4):
    envs.append(gym.make('SimpleDriving-v0'))
alice = AliceAgent()
bob = BobAgent()

checkpoint_path_bob = "./checkpoints/bob22500"
checkpoint_path_alice = "./checkpoints/alice22500"

bob.TD.load(checkpoint_path_bob)
alice.TD.load(checkpoint_path_alice)

old_alice = [deepcopy(alice)]

action_scale = np.array([0.5,0.6])
action_add = np.array([1,0])

for episode in range(EPISODES):
    alice_reward = 0
    bob_reward = 0
    jobs = [pool.apply_async(run_agents,(alice,bob,envs[i])) for i in range(4)]
    agent_rewards = np.array([j.get() for j in jobs]).sum(axis = 0)
    # with mp.Pool(4) as P:
        # print(P.map(f,[(i,i+1) for i in range(4)]))
        # agent_rewards = P.map(run_agents,[(alice,bob,envs[i]) for i in range(4)])
        # print(agent_rewards)
    # print(bob.memory)
    # print(alice.memory)
    alice_actor_error_ , alice_critic_error_ = alice.learn(learn_steps_multiplier = 4)
    bob_actor_error_ , bob_critic_error_ = bob.learn(alice, learn_steps_multiplier = 4)


    ### RUN OLD ALICE

    if episode%100 == 0:
        alice_ = deepcopy(alice)
        alice_.memory = []    
        if len(old_alice)<MAX_OLD_ALICE_LENGTH:
            old_alice.append(deepcopy(alice_))
        else:
            r = np.random.uniform(0,1)
            if r<0.3:
                old_alice[np.random.randint(0, MAX_OLD_ALICE_LENGTH)] = deepcopy(alice_)
        for j in range(2):
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
                while reward>=0:
                    action = alice_old.TD.select_action(state,gridmap) + np.random.normal(0, SIGMA, (1,2))
                    action = action * action_scale + action_add
                    next_gridmap ,next_state, reward, done,_ = env.step(*action)
                    state = deepcopy(next_state)
                    gridmap = deepcopy(next_gridmap)

                if reward != -3:
                    goal = deepcopy(state[:2])
                    gridmap, state = env.reset(goal, spawn_position, spawn_orientation, agent = "bob")
                    done = False
                    while not done:
                        # print(goal, state)
                        action = bob.TD.select_action(np.hstack((state, goal)), gridmap) + np.random.normal(0, SIGMA, (1,2))
                        action = action * action_scale + action_add
                        next_gridmap, next_state, reward, done,_ = env.step(*action, 1.5, agent = "bob")
                        reward = reward if (reward == -3 or reward>0) else 0
                        bob.memorize(gridmap, state, *action, reward, next_gridmap, next_state, goal)
                        state = deepcopy(next_state)
                        gridmap = deepcopy(next_gridmap)
                        # time.sleep(0.01)

                    if reward>0:
                        bob.memory[-1][3] = 5
                
                bob_actor_error_ , bob_critic_error_ = bob.learn(alice)
            


        # print("episode", episode,"part:", i, "-- Alice error:", alice_error, "-- Bob error:", bob_error)
    print("rewards episode", episode, "alice:", agent_rewards[0], "bob:", agent_rewards[1])
    if episode % 500 == 0:
        # alice.save_model('checkpoints/alice'+str(episode)+".pt")
        # bob.save_model('checkpoints/bob'+str(episode)+".pt")
        alice.TD.save('checkpoints/alice'+str(episode))
        bob.TD.save('checkpoints/bob'+str(episode))
    

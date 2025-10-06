import numpy as np
from copy import deepcopy

def f(x):
  return x[0]*x[0]+x[1]*x[1]

def run_agents(alice, bob, env):
    # run Alice
    alice_reward = 0
    bob_reward = 0

    spawn_position = np.random.uniform(-10,10, (3,))
    spawn_orientation = np.random.uniform(-1,1, (4,))
    spawn_position[2] = 0.5
    gridmap, state = env.reset(12*np.ones((2,)),spawn_position, spawn_orientation)
    reward = 0
    while reward>=0:
        action = alice.TD.select_action(state, gridmap) + np.random.normal(0, SIGMA, (1,2))
        action = action * action_scale + action_add
        next_gridmap, next_state, reward, done,_ = env.step(*action)
        alice.memorize(gridmap, state, *action, reward, next_gridmap, next_state)
        state = deepcopy(next_state)
        gridmap = deepcopy(next_gridmap)
    #run bob if alice sets valid goal
    if reward != -3:
        goal = deepcopy(state[:2])
        gridmap, state = env.reset(goal, spawn_position,spawn_orientation, agent = "bob")
        done = False
        while not done:
            action = bob.TD.select_action(np.hstack((state, goal)),gridmap) + np.random.normal(0, SIGMA, (1,2))
            action = action * action_scale + action_add
            next_gridmap, next_state, reward, done,_ = env.step(*action, 1.5, agent = "bob")
            reward = reward if (reward == -3 or reward>0) else 0
            bob.memorize(gridmap, state, *action, reward, next_gridmap, next_state, goal)
            state = deepcopy(next_state)
            gridmap = deepcopy(next_gridmap)
        if reward>0:
            bob.memory[-1][3] = 5
            bob_reward = bob_reward+5
            alice.memory[-1][3] = -3
            alice_reward = alice_reward - 3
        else:
            alice.memory[-1][3] = 5
            alice_reward = alice_reward+5
    return np.array([alice_reward,bob_reward])

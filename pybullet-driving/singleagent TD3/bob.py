import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
from TD3 import *
import time
EPS_START = 0.7  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 5000  # e-greedy threshold decay
GAMMA = 0.9  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
BATCH_SIZE = 25  # Q-learning batch size

device = torch.device('cuda')

class BobAgent():
    def __init__(self, action_scale, action_add):
        self.TD = TD3(9, 2, 1, 3, action_scale, action_add)
        self.steps_done = 0
        self.num_train_steps = 2
        self.memory = deque(maxlen=10000)

    def memorize(self, gridmap, state, action, reward, next_gridmap, next_state, goal, start,not_done):
        self.memory.append([gridmap,
                            np.hstack((state, goal)),
                            action,
                            reward,
                            next_gridmap,
                            np.hstack((next_state, goal)),
                            start,not_done])

    def learn(self, learn_steps_multiplier = 1):
        # tic = time.time()
        if len(self.memory) < BATCH_SIZE:
            return 0, 0
        else:
            err_actor = 0; err_critic = 0
            for i in range(self.num_train_steps*learn_steps_multiplier):
                err_actor_, err_critic_ = self.TD.train(self.memory, BATCH_SIZE, agent = "alice")
                err_actor += err_actor_; err_critic += err_critic_
            # print("bob learn time", time.time()-tic)
            return err_actor, err_critic

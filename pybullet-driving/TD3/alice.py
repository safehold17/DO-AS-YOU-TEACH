import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
import time
from TD3 import *

EPS_START = 0.7  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 5000  # e-greedy threshold decay
GAMMA = 0.9  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
BATCH_SIZE = 256  # Q-learning batch size

device = torch.device('cuda')

class AliceAgent():
    def __init__(self, action_scale, action_add):
        self.TD = TD3(7 ,2, 1, 2, action_scale, action_add)
        self.steps_done = 0
        self.num_train_steps = 2
        self.memory = deque(maxlen=50000)

    def memorize(self, gridmap, state, action, reward, next_gridmap, next_state, not_done):
        self.memory.append([gridmap,
                            state,
                            action,
                            reward,
                            next_gridmap,
                            next_state,
                            not_done])

    def learn(self, learn_steps_multiplier = 1):
        # tic = time.time()

        if len(self.memory) < BATCH_SIZE:
            return 0, 0
        else:
            err_actor = 0; err_critic = 0
            for i in range(self.num_train_steps*learn_steps_multiplier):
                err_actor_, err_critic_ = self.TD.train(self.memory, BATCH_SIZE)
                err_actor += err_actor_; err_critic += err_critic_
            # print("alice learn time", time.time()-tic)
            return err_actor, err_critic

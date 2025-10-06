import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, num_init_layers, action_scale, action_add):
        super(Actor, self).__init__()
        self.action_scale = torch.FloatTensor(action_scale).to(device) 
        self.action_add = torch.FloatTensor(action_add).to(device)

        self.cnnl1 = nn.Sequential(
            nn.Conv2d(num_init_layers,5, (4,4)),              
            nn.LeakyReLU(),
            nn.Conv2d(5,10, (5,5), stride = 2),  
            nn.LeakyReLU(),
            nn.Conv2d(10,20, (5,5), stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(20,20, (4,4), stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(20,20, (3,3)),
            nn.LeakyReLU()
        )

        self.cnnfc1 = nn.Linear(320, 100)

        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, action_dim)

        # self.cnnfc1 = nn.Linear(1620, 200)

        # self.l1 = nn.Linear(state_dim, 200)

        # self.l3 = nn.Linear(400, 100)
        # self.l4 = nn.Linear(100, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state, grid):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        grid = self.cnnl1(grid)
        x=grid.view(-1,grid.size(1) * grid.size(2) * grid.size(3))
        x = F.relu(self.cnnfc1(x))

        a = torch.cat((a,x),1)
        a = F.relu(self.l3(a))
        a = self.l4(a)

        return self.action_scale * torch.tanh(a) + self.action_add


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_init_layers):
        super(Critic, self).__init__()

        # Q1 architecture

        self.cnnl1 = nn.Sequential(
            nn.Conv2d(num_init_layers,5, (4,4)),              
            nn.LeakyReLU(),
            nn.Conv2d(5,10, (5,5), stride = 2),  
            nn.LeakyReLU(),
            nn.Conv2d(10,20, (5,5), stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(20,20, (4,4), stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(20,20, (3,3)),
            nn.LeakyReLU()
        )

        self.cnnfc1 = nn.Linear(320, 100)

        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, 1)

        # self.cnnl1 = nn.Sequential(
        #     nn.Conv2d(num_init_layers,5, (3,3), padding = 1),              
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d((3,3), stride = 2),
        #     nn.Conv2d(5,5, (3,3), padding = 1),  
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d((3,3), stride = 2)
        # )

        # self.cnnfc1 = nn.Linear(1620, 200)

        # self.l1 = nn.Linear(state_dim + action_dim, 200)

        # self.l3 = nn.Linear(400, 100)
        # self.l4 = nn.Linear(100, 1)



        # Q2 architecture
        self.cnnl2 = nn.Sequential(
            nn.Conv2d(num_init_layers,5, (4,4)),              
            nn.LeakyReLU(),
            nn.Conv2d(5,10, (5,5), stride = 2),  
            nn.LeakyReLU(),
            nn.Conv2d(10,20, (5,5), stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(20,20, (4,4), stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(20,20, (3,3)),
            nn.LeakyReLU()
        )

        self.cnnfc2 = nn.Linear(320, 100)

        self.l5 = nn.Linear(state_dim + action_dim, 100)
        self.l6 = nn.Linear(100, 100)
        self.l7 = nn.Linear(200, 100)
        self.l8 = nn.Linear(100, 1)


    def forward(self, state, action, grid):
        sa = torch.cat([state, action], 1)
        # print(sa.shape)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        grid1 = self.cnnl1(grid)
        x1=grid1.view(-1,grid1.size(1) * grid1.size(2) * grid1.size(3))
        x1 = F.relu(self.cnnfc1(x1))
        q1 = torch.cat((q1,x1),1)
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)


        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        grid2 = self.cnnl2(grid)
        x2=grid2.view(-1,grid2.size(1) * grid2.size(2) * grid2.size(3))
        x2 = F.relu(self.cnnfc2(x2))
        q2 = torch.cat((q2,x2),1)
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2


    def Q1(self, state, action, grid):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        grid1 = self.cnnl1(grid)
        x1=grid1.view(-1,grid1.size(1) * grid1.size(2) * grid1.size(3))
        x1 = F.relu(self.cnnfc1(x1))
        q1 = torch.cat((q1,x1),1)
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        num_init_layers,
        action_scale, action_add,
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.2,
        policy_freq=2
    ):
        self.actor = Actor(state_dim, action_dim, max_action, num_init_layers, action_scale, action_add).to(device)    
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, num_init_layers).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
    

    def select_action(self, state,grid):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        grid = torch.transpose(torch.FloatTensor(grid).to(device).unsqueeze_(0), 1,3)
        return self.actor(state, grid).cpu().data.numpy().flatten()
    

    def train(self, replay_buffer, batch_size=256, alice = None, agent = "alice"):
        self.total_it += 1

        # Sample replay buffer
        # tic = time.time()
        batch = random.sample(replay_buffer, batch_size)
        grid, state, action, reward, next_grid, next_state, not_done = zip(*batch)
        state = torch.FloatTensor(state).to(device)
        # grid = torch.FloatTensor(grid).to(device)
        grid = torch.from_numpy(np.asarray(grid)).to(device).float()
        grid = torch.transpose(grid, 1,3)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(not_done).to(device)
        not_done = not_done.reshape([not_done.shape[0],1])
        # next_grid = torch.transpose(torch.FloatTensor(next_grid).to(device), 1,3)
        # tic = time.time()
        next_grid = torch.transpose(torch.from_numpy(np.asarray(next_grid)).to(device).float(), 1,3)
        # print("sampling batch time", time.time()-tic)
        next_state = torch.FloatTensor(next_state).to(device)
        # print("creating batch time", time.time()-tic)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state, next_grid) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_grid)
            target_Q = torch.min(target_Q1, target_Q2)
            # print(target_Q.shape, not_done.shape)
            target_Q = reward.reshape(target_Q.shape) + not_done * self.discount * target_Q
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, grid)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actions = self.actor(state, grid)
            actor_loss = -self.critic.Q1(state, actions, grid).mean()
            if agent == "bob":
                # print(state.shape)
                alice_actions = alice.TD.actor(state[:,:7], grid[:,:2,:,:])
                actor_loss = actor_loss + 0.2*F.mse_loss(actions, alice_actions).mean()
            

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
            return actor_loss.item(), critic_loss.item()/len(batch)
        return 0, critic_loss.item()/len(batch)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.actor.state_dict(), filename + "_actor.pt")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt",map_location=torch.device('cpu')))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pt", map_location=torch.device('cpu')))
        self.actor_target = copy.deepcopy(self.actor)

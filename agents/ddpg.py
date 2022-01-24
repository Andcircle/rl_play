
import os
from overrides import overrides
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks.mlp import MLP, Actor, Critic
from memories.replay_memory import ReplayMemory
from agents.agent import Agent
from agents.noise import OUNoise

import wandb

class DDPGAgent(Agent):
    def __init__(self, env, memory, config, batch_size, actor_layers, critic_layers, actor_lr=1e-4, critic_lr=1e-3, activation='tanh', gamma=0.99, tau=1e-2, **_):
        # Params
        self.batch_size = batch_size
        
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        
        self.memory = memory
        self.noise = OUNoise(env.action_space, **config)

        # Networks
        self.actor = MLP(self.num_states, actor_layers, self.num_actions, actor_lr, activation)
        self.actor_target = MLP(self.num_states, actor_layers, self.num_actions, actor_lr, activation)
        self.critic = MLP(self.num_states + self.num_actions, critic_layers, self.num_actions, critic_lr)
        self.critic_target = MLP(self.num_states + self.num_actions, critic_layers, self.num_actions, critic_lr)

        wandb.watch(self.actor, log='all', log_graph=True, idx=0)
        wandb.watch(self.critic, log='all', log_graph=True, idx=1)
        
        self.update_target_network()
        
        # Training        
        self.critic_criterion  = nn.MSELoss()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    @overrides
    def reset(self):
        self.noise.reset()
        
        
    @overrides
    def get_action(self, state, step):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor.forward(state)
        action = action.cpu().detach().numpy()[0,0]
        return self.noise.get_action(action, step)
        # return action
    
    
    @overrides
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward() 
        self.critic.optimizer.step()
        
        self.update_target_network(self.tau)
        

    def update_target_network(self, tau=1):
        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
    
            
    @overrides
    def save(self, save_dir, epoch):
        save_path = '{}/actor_{}.pt'.format(save_dir, epoch)
        self.actor.save_checkpoint(save_path)
        wandb.save(save_path)
        
        save_path = '{}/actor_target_{}.pt'.format(save_dir, epoch)
        self.actor_target.save_checkpoint(save_path)
        wandb.save(save_path)
        
        save_path = '{}/critic_{}.pt'.format(save_dir, epoch)
        self.critic.save_checkpoint(save_path)
        wandb.save(save_path)
        
        save_path = '{}/critic_target_{}.pt'.format(save_dir, epoch)
        self.critic_target.save_checkpoint(save_path)
        wandb.save(save_path)
        
    
    @overrides
    def load(self, load_dir, epoch):
        load_path = '{}/actor_{}.pt'.format(load_dir, epoch)
        self.actor.load_state_dict(torch.load(load_path))
        
        load_path = '{}/actor_target_{}.pt'.format(load_dir, epoch)
        self.actor_target.load_state_dict(torch.load(load_path))
        
        load_path = '{}/critic_{}.pt'.format(load_dir, epoch)
        self.critic.load_state_dict(torch.load(load_path))
        
        load_path = '{}/critic_target_{}.pt'.format(load_dir, epoch)
        self.critic_target.load_state_dict(torch.load(load_path))
        
# Code source: pytorch tutorial

import copy
import numpy as np
import torch
import random
import os
from prolonged_envs.algos.networks import DQN_Net
from collections import namedtuple, deque

Transition = namedtuple('Transition',
						('state', 'action', 'reward', 'next_state', 'not_done'))

class ReplayMemory(object):

	def __init__(self, capacity, state_dim, device):
		self.memory = deque([],maxlen=capacity)
		self.state_dim = state_dim
		self.device = device

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		# return random.sample(self.memory, batch_size)
		transitions = random.sample (self.memory, batch_size)
		batch = Transition (*zip (*transitions))
		
		states = torch.cat (batch.state).reshape (-1, self.state_dim).to (self.device)
		actions = torch.cat (batch.action).reshape (-1, 1).to (self.device)
		rewards = torch.cat (batch.reward).reshape (-1, 1).to (self.device)
		next_states = torch.cat (batch.next_state).reshape (-1, self.state_dim).to (self.device)
		not_dones = torch.cat (batch.not_done).reshape (-1, 1).to (self.device)
		
		return states, actions, rewards, next_states, not_dones

	def __len__(self):
		return len(self.memory)


class DQN(object):
	def __init__(self,
				 n_actions,
				 policy_net,
				 lr,
				 discount,
				 replay_buffer,
				 batch_size,
				 target_update_freq,
				 device):
		self.n_actions = n_actions
		self.gamma = discount
		self.target_update_freq = target_update_freq
		
		self.policy_net = policy_net
		self.target_net = copy.deepcopy(self.policy_net).to(device)
		self.target_net.load_state_dict (self.policy_net.state_dict ())
		self.target_net.eval ()
		
		self.optimizer = torch.optim.Adam (self.policy_net.parameters (), lr=lr)
		self.criterion = torch.nn.MSELoss ()
		
		self.memory = replay_buffer
		self.batch_size = batch_size
		
		self.device = device
		self.counter = 0
		
	def select_action (self, state, eps):
		if np.random.uniform () > eps:
			with torch.no_grad ():
				return self.policy_net (state).argmax().item()
		else:
			return np.random.randint (self.n_actions)
	
	def train(self):
		states, actions, rewards, next_states, not_done = self.memory.sample(self.batch_size)
		
		# Compute Q(s_t, a)
		q_values = self.policy_net(states).gather (1, actions)
		
		# Compute V(s_{t+1}) for all next states.
		next_state_values = self.target_net(next_states).max(dim=1)[ 0 ].detach ().reshape (-1, 1)
		
		# Compute the expected Q values
		expected_q_values = rewards + (not_done * self.gamma * next_state_values)
		
		# Compute loss
		loss = self.criterion(q_values, expected_q_values)
		
		# Optimize the model
		self.optimizer.zero_grad ()
		loss.backward ()
		
		for param in self.policy_net.parameters():
			param.grad.data.clamp_ (-1, 1)
			
		self.optimizer.step()
		
		# Updating target net once every episode
		if self.counter % self.target_update_freq == 0:
			self.target_net.load_state_dict (self.policy_net.state_dict())
			
		self.counter += 1
		
	def save (self, path):
		torch.save (self.policy_net.state_dict (), os.path.join(path, 'policy.pth'))
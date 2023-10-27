# Implementation:
# https://mlpeschl.com/post/tiny_adrqn/
# https://colab.research.google.com/github/mlpeschl/stat-notebooks/blob/master/Model_Free_RL/ADRQN.ipynb

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class ExpBuffer ():
	def __init__ ( self, max_storage, sample_length, device):
		self.max_storage = max_storage
		self.sample_length = sample_length
		self.counter = -1
		self.filled = -1
		self.storage = [ 0 for i in range (max_storage) ]
		self.device = device
	
	def write_tuple ( self, aoarod ):
		if self.counter < self.max_storage - 1:
			self.counter += 1
		if self.filled < self.max_storage:
			self.filled += 1
		else:
			self.counter = 0
		self.storage[ self.counter ] = aoarod
	
	def sample ( self, batch_size ):
		# Returns sizes of (batch_size, seq_len, *) depending on action/observation/return/done
		seq_len = self.sample_length
		last_actions = [ ]
		last_observations = [ ]
		actions = [ ]
		rewards = [ ]
		observations = [ ]
		dones = [ ]
		
		for i in range (batch_size):
			if self.filled - seq_len < 0:
				raise Exception ("Reduce seq_len or increase exploration at start.")
			start_idx = np.random.randint (self.filled - seq_len)
			# print(self.filled)
			# print(start_idx)
			last_act, last_obs, act, rew, obs, done = zip (*self.storage[ start_idx:start_idx + seq_len ])
			last_actions.append (list (last_act))
			last_observations.append (last_obs)
			actions.append (list (act))
			rewards.append (list (rew))
			observations.append (list (obs))
			dones.append (list (done))
		
		return torch.tensor (np.array(last_actions)), torch.tensor (np.array(last_observations),
																	dtype=torch.float32).to(self.device), \
			   torch.tensor (np.array(actions)).to(self.device), \
			   torch.tensor (np.array(rewards)).float ().to(self.device), \
			   torch.tensor (np.array(observations), dtype=torch.float32).to(self.device), \
			   torch.tensor (np.array(dones)).to(self.device)



class ADRQN_Diabetes(nn.Module):
	def __init__ ( self, n_actions, state_size, action_embedding_size, state_embedding_size, n_hidden):
		super (ADRQN_Diabetes, self).__init__()
		self.n_actions = n_actions
		# self.embedding_size = action_embedding_size
		self.state_embedder = nn.Linear (state_size, state_embedding_size)
		self.action_embedder = nn.Linear (n_actions, action_embedding_size)
		
		self.lstm = nn.LSTM (input_size=state_embedding_size + action_embedding_size, hidden_size=n_hidden, batch_first=True)
		
		self.out_layer = nn.Linear (n_hidden, n_actions)
	
	def forward ( self, observation, action, hidden=None ):
		# Takes observations with shape (batch_size, seq_len, obs_dim)
		# Takes one_hot actions with shape (batch_size, seq_len, n_actions)
		state_embedding = self.state_embedder (observation)
		action_embedding = self.action_embedder (action)
		lstm_input = torch.cat ([ state_embedding, action_embedding ], dim=-1)
		if hidden is not None:
			lstm_out, hidden_out = self.lstm (lstm_input, hidden)
		else:
			lstm_out, hidden_out = self.lstm (lstm_input)
		
		q_values = self.out_layer (lstm_out)
		return q_values, hidden_out
	
	def act ( self, observation, last_action, epsilon, hidden=None ):
		q_values, hidden_out = self.forward (observation, last_action, hidden)
		if np.random.uniform () > epsilon:
			action = torch.argmax (q_values).item ()
		else:
			action = np.random.randint (self.n_actions)
		return action, hidden_out



import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_Net (nn.Module):
	# Very simple fully connected network with relu activations
	def __init__ ( self, n_actions, state_size, state_embedding_size,
				   n_hidden ):
		super (DQN_Net, self).__init__ ()
		self.state_embedder = nn.Linear (state_size, state_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear (state_embedding_size, n_hidden)
		# self.obs_layer2 = nn.Linear (n_hidden, 2 * n_hidden)
		self.obs_layerout = nn.Linear (n_hidden, n_actions)
	
	def forward ( self, observation ):
		state_embedding = self.state_embedder (observation)
		observation = F.relu (self.obs_layer (state_embedding))
		# observation = F.relu (self.obs_layer2 (observation))
		return self.obs_layerout (observation)


class DQN_Diabetes_Net(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__ ( self, n_actions, state_size, state_embedding_size, n_hidden):
		super (DQN_Diabetes_Net, self).__init__ ()
		self.state_embedder = nn.Linear (state_size, state_embedding_size)
		self.obs_layer = nn.Linear (state_embedding_size, n_hidden)
		self.obs_layerout = nn.Linear (n_hidden, n_actions)

	def forward ( self, observation ):
		observation = self.state_embedder (observation)
		observation = F.relu (self.obs_layer (observation))
		return self.obs_layerout (observation)


class EFFDQN_Diabetes_Real_Net (nn.Module):
	# Very simple fully connected network with relu activations
	def __init__ ( self, n_actions, action_size, state_size, action_embedding_size, state_embedding_size, n_hidden):
		super (EFFDQN_Diabetes_Real_Net, self).__init__ ()
		self.action_embedder = nn.Linear (action_size, action_embedding_size)
		self.state_embedder = nn.Linear (state_size, state_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear (state_embedding_size + action_embedding_size, n_hidden)
		self.obs_layerout = nn.Linear (n_hidden, n_actions)
	
	def forward ( self, observation):
		state, eff_action = torch.split (observation, 1, dim=1)
		state_embedding = self.state_embedder(state)
		action_embedding = self.action_embedder(eff_action)
		observation = torch.cat ((state_embedding, action_embedding), dim=1)
		observation = F.relu (self.obs_layer (observation))
		return self.obs_layerout (observation)
	
	
class EFFDQN_Real_Net(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__ ( self, n_actions, action_size, state_size, action_embedding_size, state_embedding_size,
				   n_hidden ):
		super (EFFDQN_Real_Net, self).__init__ ()
		self.action_embedder = nn.Linear (action_size, action_embedding_size)
		self.state_embedder = nn.Linear (state_size, state_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear (state_embedding_size + action_embedding_size, n_hidden)
		# self.obs_layer2 = nn.Linear (n_hidden, 2 * n_hidden)
		self.obs_layerout = nn.Linear (n_hidden, n_actions)
	
	def forward ( self, observation ):
		state, eff_action = torch.split (observation, 1, dim=1)
		state_embedding = self.state_embedder (state)
		action_embedding = self.action_embedder (eff_action)
		observation = torch.cat ((state_embedding, action_embedding), dim=1)
		observation = F.relu (self.obs_layer (observation))
		# observation = F.relu (self.obs_layer2 (observation))
		return self.obs_layerout (observation)
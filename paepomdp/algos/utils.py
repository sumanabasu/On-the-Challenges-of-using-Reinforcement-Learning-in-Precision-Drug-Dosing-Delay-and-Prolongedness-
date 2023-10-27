import numpy as np
import gym
import torch
import os
import pickle
import torch.nn.functional as F
# def get_discrete_position ( position_space, position ):
# 	return int (np.digitize (position, position_space) - 1)


def make_epsilon_greedy_policy ( Q, epsilon, nA ):
	"""
	Creates an epsilon-greedy policy based on a given Q-function and epsilon.

	Args:
		Q: A dictionary that maps from state -> action-values.
			Each value is a numpy array of length nA (see below)
		epsilon: The probability to select a random action. Float between 0 and 1.
		nA: Number of actions in the environment.

	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length nA.

	"""
	
	def policy_fn ( observation, epsilon ):
		A = np.ones (nA, dtype=float) * epsilon / nA
		best_action = np.argmax (Q[ observation ])
		A[ best_action ] += (1.0 - epsilon)
		return A
	
	return policy_fn


def softmax_policy(Q, temperature):
	num = np.exp(Q / temperature)
	probs = (num/np.sum(num))
	return probs


def effective_action(lambdaa, curr_action, prev_effective_action, max_action=None):
	effective_action = lambdaa * prev_effective_action + curr_action
	if max_action is None:
		return effective_action
	return min(np.array([max_action]), effective_action)

	
def augmented_state_encoding(state, effective_action):
	return np.concatenate((state, effective_action))


class PostionWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		super(PostionWrapper, self).__init__(env)
		self.observation_space = gym.spaces.Box(low=self.env.min_position, high=self.env.max_position, shape=[1])
		
	def observation(self, observation):
		return np.array([observation[0]])
	
class DiscretizePositionWrapper(gym.ObservationWrapper):
	def __init__(self, env, spacing=0.05, precision=2):
		super(DiscretizePositionWrapper, self).__init__(env)
		# self.position_space = np.linspace (env.observation_space.low[ 0 ], env.observation_space.high[ 0 ],
		# 						  int (env.observation_space.high[ 0 ] - env.observation_space.low[ 0 ] + 1))
		self.position_space = np.linspace(env.observation_space.low[ 0 ], env.observation_space.high[ 0 ],
                              int ((env.observation_space.high[ 0 ] - env.observation_space.low[ 0 ]) /spacing) + 1)
		
		self.position_space = np.round(self.position_space, precision)
		self.precision = precision
		
	def observation(self, observation):
		return round(self.position_space[int(np.digitize(observation, self.position_space) - 1)], self.precision)
	
	
def moving_average(x, w=100):
	return np.convolve(x, np.ones(w), 'valid') / w


def set_seed ( env, seed ):
	# Set seeds
	env.seed (seed)
	env.action_space.seed (seed)
	torch.manual_seed (seed)
	np.random.seed (seed)
	
	return env

def savePickle(location, file, data):
	pkl_file = open(os.path.join(location, file), 'wb')
	pickle.dump(data, pkl_file)
	pkl_file.close()
	
	
def loadPickle(location, file):
	pkl_file = open (os.path.join(location, file), 'rb')
	data = pickle.load (pkl_file)
	return data


def effective_action_vector(lambdaa, n_actions, action, prev_effective_action):
		"""
		:param action: 1-D action value (float or int)
		:param prev_effective_action: k-hot encoded and lambda decayed previous effective action
		:return:
		"""
		action = F.one_hot (torch.tensor (action), n_actions)
		effective_action = (lambdaa * prev_effective_action + action).ravel()
		idx = torch.nonzero(effective_action)
		# idx = idx.ravel()
		softmax = F.softmax(effective_action[ idx ], dim=0)
		effective_action[idx] = softmax
		return effective_action
		# return F.softmax(effective_action, dim=1)
		# return effective_action / max(effective_action)

def augmented_state( state, effective_action):
	# return torch.cat ((torch.tensor (state.clone().detach()).float().view (1, -1),
	# 			effective_action.clone().detach().view (1, -1)),
	# 		   dim=-1)
	s = state.clone ().detach ()
	a = effective_action.clone ().detach ()
	return torch.cat ((s.float ().view (1, -1), a.view (1, -1)), dim=-1)
	# return (state, effective_action)
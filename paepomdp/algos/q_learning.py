#https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb

import itertools
import math
import numpy as np
import sys

from collections import defaultdict
from prolonged_envs.algos.utils import make_epsilon_greedy_policy

def q_learning(env, seed, init_pos, num_episodes, eps_start, eps_end, eps_decay, discount_factor=1.0, alpha=0.5):
	"""
	Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
	while following an epsilon-greedy policy

	Args:
		env: OpenAI environment.
		num_episodes: Number of episodes to run for.
		discount_factor: Gamma discount factor.
		alpha: TD learning rate.
		epsilon: Chance to sample a random action. Float between 0 and 1.

	Returns:
		A tuple (Q, episode_lengths).
		Q is the optimal action-value function, a dictionary mapping state -> action values.
		stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	"""
	
	env.seed (seed)
	np.random.seed(seed)
	
	q_returns = np.zeros((num_episodes))
	q_episode_lengths = np.zeros ((num_episodes))
	start_states = []
	
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	
	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, eps_start, env.action_space.n)
	
	for i_episode in range (num_episodes):
		eps = eps_end + (eps_start - eps_end) * math.exp ((-1 * i_episode) / eps_decay)
		
		# Reset the environment and pick the first action
		state = env.reset (init_pos=init_pos)
		start_states.append(state)
		
		# One step in the environment
		# total_reward = 0.0
		for t in itertools.count():
			
			# Take a step
			action_probs = policy(state, eps)
			action = np.random.choice(np.arange (len (action_probs)), p=action_probs)
			next_state, reward, done, _ = env.step (np.array ([ action ]))
			
			# if i_episode > 49000:
			# 	print('At state:', state, ' Action:', action,' taken at t = ', t, 'reward:', reward)
			
			# Update statistics
			q_returns[ i_episode ] += reward
			q_episode_lengths[ i_episode ] = t
			
			# TD Update
			best_next_action = np.argmax (Q[ next_state ])
			td_target = reward + discount_factor * Q[ next_state ][ best_next_action ]
			td_delta = td_target - Q[ state ][ action ]
			Q[ state ][ action ] += alpha * td_delta
			
			if done:
				# if i_episode > 40000:
				# 	print('return:',q_returns[ i_episode ])
				break
			
			state = next_state
			
		# Print out which episode we're on, useful for debugging.
		if (i_episode + 1) % 100 == 0:
			print ("\rEpisode {}/{}.".format (i_episode + 1, num_episodes), end="")
			sys.stdout.flush ()
	
	return Q, np.array(start_states), q_returns, q_episode_lengths
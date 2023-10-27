import sys
import itertools
import math
import numpy as np

from collections import defaultdict
from code.algos.utils import make_epsilon_greedy_policy

def effective_action(max_action, lambdaa, curr_action, prev_effective_action, precision, action_space):
	effective_action = lambdaa * prev_effective_action + curr_action
	effective_action = action_space[ int (np.digitize (effective_action, action_space) - 1) ]
	return min(max_action, round(effective_action, precision))


def effective_q_learning ( env, seed, num_episodes, eps_start, eps_end, eps_decay, discount_factor, alpha,
			lambdaa, init_pos):

	env.seed (seed)
	np.random.seed (seed)
	
	q_returns = np.zeros ((num_episodes))
	q_episode_lengths = np.zeros ((num_episodes))
	start_states = [ ]
	
	act_spacing = 0.05
	precision = 2
	action_space = np.linspace (0, env.action_space.n, int ((env.action_space.n - 0) / act_spacing) + 1)

	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).

	Q = defaultdict (lambda: np.zeros (env.action_space.n))
	prev_eff_action = env.min_action

	# The policy we're following
	policy = make_epsilon_greedy_policy (Q, eps_start, env.action_space.n)

	for i_episode in range (num_episodes):
		eps = eps_end + (eps_start - eps_end) * math.exp ((-1 * i_episode) / eps_decay)

		# Reset the environment and pick the first action
		state = env.reset(init_pos=init_pos)
		start_states.append (state)

		for t in itertools.count ():
			# Take a step
			aug_state = (state, prev_eff_action)
			action_probs = policy(aug_state, eps)

			action = np.random.choice(np.arange (len (action_probs)), p=action_probs)
			next_state, reward, done, _ = env.step (np.array ([ action ]))

			# Update statistics
			q_returns[ i_episode ] += reward
			q_episode_lengths[ i_episode ] = t

			# effective TD Update
			# calculate the effective action
			eff_action = effective_action(env.max_action, lambdaa, action, prev_eff_action, precision, action_space)

			# augmented state : (state, effective action)
			aug_next_state = (next_state, eff_action)

			# select the next action based on the Q value of the augmented state
			best_next_action = np.argmax (Q[ aug_next_state ])

			# TD update on the new MDP i.e. the augmented state
			td_target = reward + discount_factor * Q[ aug_next_state ][ best_next_action ]
			td_delta = td_target - Q[ aug_state ][ action ]
			Q[ aug_state ][ action ] += alpha * td_delta

			if done:
				break

			state = next_state
			prev_eff_action = eff_action

		# Print out which episode we're on, useful for debugging.
		if (i_episode + 1) % 100 == 0:
			print ("\rEpisode {}/{}.".format (i_episode + 1, num_episodes), end="")
			sys.stdout.flush ()

	return Q, np.array(start_states), q_returns, q_episode_lengths
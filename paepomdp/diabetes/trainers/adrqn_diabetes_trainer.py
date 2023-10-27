import copy
import numpy as np
import gym
import torch
import torch.nn.functional as F
from itertools import count
import math
from collections import defaultdict
from code.algos.ADRQN import ADRQN, ADRQN_Diabetes, ExpBuffer
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from code.diabetes.helpers.utils import register_single_patient_env, DiscretizeActionWrapper, \
	save_learning_metrics

def train(env,
		  dir_,
		  state_size,
		  n_actions,
		  M_episodes,
		  replay_buffer_size,
		  batch_size,
		  eps_start,
		  eps_end,
		  gamma,
		  state_embedding_size,
		  action_embedding_size,
		  n_hidden,
		  sample_length,
		  learning_rate,
		  eps_decay,
		  EXPLORE,
		  seed,
		  hyperglycemic_BG,
		  hypoglycemic_BG):
	
	device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
	print('\n device:', device)
	metrics = defaultdict (list)
	checkpoint_freq = 1000
	
	# Set seeds
	seed = seed
	env.seed (seed)
	# env.action_space.seed (seed)
	torch.manual_seed (seed)
	np.random.seed (seed)
	
	adrqn = ADRQN_Diabetes(n_actions,
						   state_size,
						   action_embedding_size,
						   state_embedding_size,
						   n_hidden).to(device)
	# adrqn_target = ADRQN_Diabetes(n_actions, state_size, action_embedding_size,
	# 					   state_embedding_size, n_hidden).to(device)
	adrqn_target = copy.deepcopy(adrqn).to(device)
	adrqn_target.load_state_dict (adrqn.state_dict ())
	
	print ('Network:', adrqn, file=open (os.path.join (dir_, 'architecture.txt'), 'a'))
	
	optimizer = torch.optim.Adam (adrqn.parameters (), lr=learning_rate)
	replay_buffer = ExpBuffer (replay_buffer_size, sample_length, device=device)
	eps = eps_start
	
	for i_episode in range (M_episodes):
		done = False
		hidden = None
		last_action = 0
		current_return = 0
		last_observation = env.reset ()
		
		while done == False:
			last_action = torch.tensor(last_action)
			
			action, hidden = adrqn.act (torch.tensor (last_observation).float ().view (1, 1, -1).to (device),
										F.one_hot (torch.tensor (last_action.item()), n_actions).view (1, 1, -1).float ().to (device),
										hidden=hidden, epsilon=eps)
			
			# if last_observation[0] < 150:
			# 	action = 0
			
			observation, reward, done, info = env.step(np.array([action]))
			
			current_return += reward
			replay_buffer.write_tuple ((last_action, last_observation, action, reward, observation, done))
			
			last_action = action
			last_observation = observation
			
			# Updating Networks
			if i_episode > EXPLORE:
				eps = eps_end + (eps_start - eps_end) * math.exp ((-1 * (i_episode - EXPLORE)) / eps_decay)
				
				last_actions, last_observations, actions, rewards, observations, dones = replay_buffer.sample (batch_size)
				q_values, _ = adrqn.forward (last_observations, F.one_hot (last_actions, n_actions).float ().to(device))
				q_values = torch.gather (q_values, -1, actions.unsqueeze (-1)).squeeze (-1)
				predicted_q_values, _ = adrqn_target.forward (observations, F.one_hot (actions, n_actions).float().to(device))
				
				target_values = rewards + (gamma * (1 - dones.float ()) * torch.max (predicted_q_values, dim=-1)[ 0 ])
				
				# Update network parameters
				optimizer.zero_grad ()
				loss = torch.nn.MSELoss () (q_values, target_values.detach ())
				loss.backward ()
				optimizer.step ()
			
			metrics[ 'action_hist' ].append (action)
		
		adrqn_target.load_state_dict (adrqn.state_dict ())
		
		hyperglycemic_zone_len = np.where (np.array (env.env.env.BG_hist) > hyperglycemic_BG)[ 0 ].shape[ 0 ]
		hypoglycemic_zone_len = np.where (np.array (env.env.env.BG_hist) < hypoglycemic_BG)[ 0 ].shape[ 0 ]
		target_zone_len = len (env.env.env.BG_hist) - (hyperglycemic_zone_len + hypoglycemic_zone_len)
			
		# save
		metrics[ 'training_reward' ].append (current_return)
		metrics[ 'hyperglycemic_BG' ].append (hyperglycemic_zone_len)
		metrics[ 'hypoglycemic_BG' ].append (hypoglycemic_zone_len)
		metrics[ 'target_BG' ].append (target_zone_len)
		
		metrics[ 'BG_hist' ].extend (env.env.env.BG_hist[:-1])
		metrics[ 'CGM_hist' ].extend (env.env.env.CGM_hist[:-1])
		metrics[ 'insulin_hist' ].extend (list (np.array (env.env.env.insulin_hist).flatten ()))
		metrics[ 'CHO_hist' ].extend (env.env.env.CHO_hist)
		metrics[ 'mortality' ].append(env.env.env.BG_hist[ -1 ])
		
		if i_episode % 1000 == 0:
			print (f"Episode: {i_episode + 1}  Reward: {current_return:.3f}")
		
		if i_episode % checkpoint_freq == 0:
			save_learning_metrics (dir_, **metrics)
			# TODO: checkpoint model
		
	save_learning_metrics (dir_, **metrics)
	torch.save(adrqn.state_dict (), dir_+'/adrqn_policy.pth')
	
	print('Done!')
	env.close ()
	
if __name__ == '__main__':
	EXPERIMENTS = "../experiments/"
	
	patient_name = 'adult#009'
	reward = 'zone_reward'
	seed = 10
	
	env_id = register_single_patient_env (patient_name,
										  reward_fun=reward,
										  seed=seed,
										  version='-v0')
	env = gym.make (env_id)
	env = DiscretizeActionWrapper (env, low=0, high=5, n_bins=6)
	
	kwargs = {
			# Fixed
			'state_size': env.observation_space.shape[ 0 ],
			'n_actions': env.num_actions,
			'M_episodes': 1000,
			'replay_buffer_size': 100000,
			'batch_size': 64,
			'eps_start': 0.9,
			'eps_end': 0.05,
			'gamma': 0.999,
			'sample_length': 5,
			'learning_rate': 0.01,
			'eps_decay'	:	10,
			'EXPLORE' : 300,
			'seed': 1,
			'hyperglycemic_BG' : 150,
			'hypoglycemic_BG' : 100,
			'n_hidden': 256,
			'action_embedding_size': 16,
			'state_embedding_size': 16
		}
	
	expt_id = datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')
	
	if not os.path.exists (EXPERIMENTS):
		os.makedirs (EXPERIMENTS)
	
	dir_ = EXPERIMENTS + expt_id
	os.makedirs (dir_)
	print (f"Created {dir_}")
	
	train(env, dir_, **kwargs)
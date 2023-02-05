import matplotlib
import torch
import gym
import math
import numpy as np
import os
import random
import time
from datetime import datetime
from collections import defaultdict
from paepomdp.algos.DQN import ReplayMemory, DQN
from paepomdp.algos.networks import DQN_Net, DQN_Diabetes_Net
from paepomdp.algos.utils import savePickle
from paepomdp.diabetes.helpers.rewards import zone_reward
from paepomdp.diabetes.helpers.utils import register_single_patient_env, DiscretizeActionWrapper, \
	save_learning_metrics


def train ( env,
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
			learning_rate,
			eps_decay,
			EXPLORE,
			TARGET_UPDATE,
			seed,
			hyperglycemic_BG,
			hypoglycemic_BG,
			n_hidden):
	
	# device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
	device = torch.device ("cuda")
	print ('device:', device)
	# writer = SummaryWriter (dir_)
	metrics = defaultdict (list)
	
	# Set seeds
	seed = seed
	env.seed (seed)
	# env.action_space.seed (seed)
	torch.manual_seed (seed)
	np.random.seed (seed)
	random.seed (seed)
	
	# Initialize Replay Buffer
	replay_buffer = ReplayMemory (replay_buffer_size, state_size, device)
	
	# Initialize DQN algorithm
	policy_net = DQN_Diabetes_Net(n_actions, state_size, state_embedding_size, n_hidden).to (device)
	
	policy = DQN (n_actions,
				  policy_net,
				  learning_rate,
				  gamma,
				  replay_buffer,
				  batch_size,
				  TARGET_UPDATE,
				  device)
	
	print ('Network:', policy.policy_net, file=open (os.path.join (dir_, 'architecture.txt'), 'a'))
	
	checkpoint_freq = 1000
	
	# Training Loop
	for i_episode in range (M_episodes):
		# Adaptive epsilon scheme
		eps = eps_end + (eps_start - eps_end) * math.exp ((-1 * i_episode) / eps_decay)
		done = False
		current_return = 0
		state = env.reset ()
		
		while done == False:
			state = torch.FloatTensor (state)
			# if state.item() < 150:
			# 	action = 0	#np.array([0])
			# else:
			action = policy.select_action (state.to (device), eps)
				
			assert env.max_action >= action >= env.min_action
			next_state, reward, done, info = env.step (np.array ([ action ]))
			done_bool = float (done)
			current_return += reward
			
			replay_buffer.push (state,  # torch.FloatTensor (state)
								torch.LongTensor ([ action ]),
								torch.FloatTensor ([ reward ]),
								torch.FloatTensor (next_state),
								torch.tensor ([ 1 - done_bool ]))
			
			state = next_state
			
			# Updating Networks
			if i_episode > EXPLORE and len (replay_buffer) > batch_size:
				policy.train ()
			
			metrics[ 'action_hist' ].append (action)
		
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
		metrics[ 'insulin_hist' ].extend(np.concatenate(env.env.env.insulin_hist).ravel().tolist())
		#(list(np.array (env.env.env.insulin_hist, dtype=float).flatten()))
		# metrics[ 'insulin_hist' ].extend (list(np.vstack(env.env.env.insulin_hist).flatten()))
		metrics[ 'CHO_hist' ].extend (env.env.env.CHO_hist)
		metrics[ 'mortality' ].append(env.env.env.BG_hist[ -1 ])
		
		# return_list.append (current_return)
		
		if i_episode % 1000 == 0:
			print (f"Episode: {i_episode + 1}  Reward: {current_return:.3f}")
		
		if i_episode % checkpoint_freq == 0:
			if dir_ is not None:
				save_learning_metrics (dir_, **metrics)
				# savePickle (dir_, 'returns.pkl', np.asarray (return_list))
				policy.save (dir_)
	if dir_ is not None:
		# savePickle (dir_, 'returns.pkl', np.asarray (return_list))
		save_learning_metrics (dir_, **metrics)
		policy.save(dir_)
	
	print ('Done!')
	
	env.close ()
	# writer.flush ()
	# writer.close ()


if __name__ == '__main__':
	EXPERIMENTS = "../../../Experiments/DQN/"
	
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
		'state_size': env.observation_space.shape[0],
		'n_actions': env.num_actions,
		'M_episodes': 10000,
		'replay_buffer_size': 1000000,
		'batch_size': 512,
		'eps_start': 0.9,
		'eps_end': 0.05,
		'gamma': 0.999,
		'learning_rate': 0.000001,
		'eps_decay': 500,
		'EXPLORE': 1000,
		'TARGET_UPDATE': 1,
		'seed': 1,
		'hyperglycemic_BG' : 150,
		'hypoglycemic_BG' : 100,
		'n_hidden' : 256,
		'state_embedding_size': 16
	}
	
	expt_id = datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')

	if not os.path.exists (EXPERIMENTS):
		os.makedirs (EXPERIMENTS)

	dir_ = EXPERIMENTS + expt_id
	os.makedirs (dir_)
	print (f"Created {dir_}")
	
	print(kwargs)
	
	train (env, dir_, **kwargs)
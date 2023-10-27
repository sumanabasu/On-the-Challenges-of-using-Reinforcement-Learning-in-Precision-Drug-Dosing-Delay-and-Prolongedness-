import gym
import math
import numpy as np
import os
import random
import torch
import argparse
from collections import defaultdict
from datetime import datetime
from paepomdp.algos.DQN import ReplayMemory, DQN
from paepomdp.algos.networks import EFFDQN_Diabetes_Real_Net
from paepomdp.diabetes.helpers.utils import register_single_patient_env, DiscretizeActionWrapper, \
	save_learning_metrics

def train(env,
		  dir_,
		  action_size,
		  state_size,
		  n_actions,
		  state_embedding_size,
		  action_embedding_size,
		  M_episodes,
		  replay_buffer_size,
		  batch_size,
		  eps_start,
		  eps_end,
		  gamma,
		  lambdaa,
		  learning_rate,
		  eps_decay,
		  EXPLORE,
		  TARGET_UPDATE,
		  seed,
		  hyperglycemic_BG,
		  hypoglycemic_BG,
		  n_hidden):
	device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")

	metrics = defaultdict (list)
	
	# Set seeds
	seed = seed
	env.seed (seed)
	torch.manual_seed (seed)
	np.random.seed (seed)
	random.seed(seed)
	
	# Initialize Replay Buffer
	replay_buffer = ReplayMemory (replay_buffer_size, state_size+action_size, device)
	policy_net = EFFDQN_Diabetes_Real_Net(n_actions, action_size,
										  state_size, action_embedding_size,
										  state_embedding_size, n_hidden).to (device)
	
	# Initialize DQN algorithm
	policy = DQN(n_actions,
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
		eff_action = env.min_action
		aug_state = torch.cat((torch.tensor([state[0]]).float().view (1, -1).to(device),
							   torch.tensor([eff_action]).float().view (1, -1).to(device)), dim=-1)
		
		while done == False:
			action = policy.select_action(aug_state, eps)
			next_state, reward, done, info = env.step(np.array([action]))
			done_bool = float (done)
			current_return += reward
			
			next_eff_action = lambdaa * eff_action + action
			
			next_aug_state = torch.cat((torch.tensor([next_state[0]]).float().view (1, -1).to(device),
										torch.tensor([next_eff_action]).float().view (1, -1).to(device)), dim=-1)
			
			replay_buffer.push (aug_state,
								torch.LongTensor ([ action ]),
								torch.FloatTensor ([ reward ]),
								next_aug_state,
								torch.tensor ([ 1 - done_bool ]))
			
			aug_state = next_aug_state
			eff_action = next_eff_action
			
			# Training
			if i_episode > EXPLORE and len(replay_buffer) > batch_size:
				policy.train()
			
			metrics[ 'action_hist' ].append (action)
		
		hyperglycemic_zone_len = np.where (np.array (env.env.env.BG_hist) > hyperglycemic_BG)[ 0 ].shape[ 0 ]
		hypoglycemic_zone_len = np.where (np.array (env.env.env.BG_hist) < hypoglycemic_BG)[ 0 ].shape[ 0 ]
		target_zone_len = len (env.env.env.BG_hist) - (hyperglycemic_zone_len + hypoglycemic_zone_len)
		
		# save
		metrics[ 'training_reward' ].append (round(current_return, 2))
		metrics[ 'hyperglycemic_BG' ].append (hyperglycemic_zone_len)
		metrics[ 'hypoglycemic_BG' ].append (hypoglycemic_zone_len)
		metrics[ 'target_BG' ].append (target_zone_len)
		
		metrics[ 'BG_hist' ].extend (env.env.env.BG_hist[:-1])
		metrics[ 'CGM_hist' ].extend (env.env.env.CGM_hist[:-1])
		metrics[ 'insulin_hist' ].extend(np.concatenate(env.env.env.insulin_hist).ravel().tolist())
		metrics[ 'CHO_hist' ].extend (env.env.env.CHO_hist)
		metrics[ 'mortality' ].append(env.env.env.BG_hist[-1])
		
		
		if i_episode % 1000 == 0:
			print (f"Episode: {i_episode + 1}  Reward: {current_return:.3f}")
			
		if i_episode % checkpoint_freq == 0:
			save_learning_metrics (dir_, **metrics)
			policy.save (dir_)
	
	save_learning_metrics (dir_, **metrics)
	policy.save (dir_)
	
	
	print ('Done!')
	
	env.close ()
	
def run(args):
	EXPERIMENTS = "../../../Experiments/EffDQN/"
	print('Training Effective DQN for patient: ', args.patient_name)

	env_id = register_single_patient_env (args.patient_name,
										  reward_fun=args.reward,
										  seed=args.seed,
										  version='-v0')
	env = gym.make (env_id)
	env = DiscretizeActionWrapper (env, low=0, high=5, n_bins=6)
	
	kwargs = {
			# Fixed
			'state_size': env.observation_space.shape[ 0 ],
			'action_size' : env.action_space.shape[0],
			'n_actions': env.num_actions,
			'state_embedding_size': 16,
			'action_embedding_size': 16,
			'M_episodes': 10000,
			'replay_buffer_size': 1000000,
			'batch_size': 512,
			'eps_start': 0.9,
			'eps_end': 0.05,
			'gamma': 0.999,
			'lambdaa': 0.95,
			'learning_rate': 0.001,
			'eps_decay': 500,
			'EXPLORE': 1000,
			'TARGET_UPDATE': 1,
			'seed': 1,
			'hyperglycemic_BG' : 150,
			'hypoglycemic_BG' : 100,
			'n_hidden' : 256
	}
	
	expt_id = datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')
	
	if not os.path.exists (EXPERIMENTS):
		os.makedirs (EXPERIMENTS)
	
	dir_ = EXPERIMENTS + expt_id
	os.makedirs (dir_)
	print (f"Created {dir_}")
	
	train (env, dir_,  **kwargs)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser (description="EffDQN argument parser")
	parser.add_argument ("--patient_name", type=str, default='adult#009', help="Name of the patient")
	parser.add_argument ("--reward", type=str, default='zone_reward', help="Reward type (default: zone_reward)")
	parser.add_argument ("--seed", type=int, default=3, help="Seed value (default: 10)")
		
	args = parser.parse_args ()
		
	run(args)
import argparse
import gym
import os
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict
from prolonged_envs.diabetes.helpers.utils import register_single_patient_env, DiscretizeActionWrapper, save_learning_metrics


def fixed_dose(env, dir_, hyperglycemic_BG, hypoglycemic_BG, M_episodes, seed):
	# Set seeds
	env.seed (seed)
	np.random.seed (seed)
	
	def act(bg):
		'''
			:return: no insulin for target and hypoglycemic zone.
					Else 1 unit higher dose of insulin for every 10 units of increase in blood glucose
		'''
		dose = 0
		
		if bg >= 190:
			dose = 5
		elif bg >= 180:
			dose = 4
		elif bg >= 170:
			dose = 3
		elif bg >= 160:
			dose = 2
		elif bg > 150:
			dose = 1
		
		return dose
	
	metrics = defaultdict (list)
	
	for i_episode in range (M_episodes):
		done = False
		current_return = 0
		state = env.reset ()
		
		while done == False:
			if state[0] < 150:
				action = 0  # np.array([0])
			else:
				action = act(state[0])
			next_state, reward, done, info = env.step (np.array ([action]))
			current_return += reward
			
			state = next_state
		
		hyperglycemic_zone_len = np.where (np.array (env.env.env.BG_hist) > hyperglycemic_BG)[ 0 ].shape[ 0 ]
		hypoglycemic_zone_len = np.where (np.array (env.env.env.BG_hist) < hypoglycemic_BG)[ 0 ].shape[ 0 ]
		target_zone_len = len (env.env.env.BG_hist) - (hyperglycemic_zone_len + hypoglycemic_zone_len)
		
		# save
		metrics[ 'training_reward' ].append (current_return)
		metrics[ 'hyperglycemic_BG' ].append (hyperglycemic_zone_len)
		metrics[ 'hypoglycemic_BG' ].append (hypoglycemic_zone_len)
		metrics[ 'target_BG' ].append (target_zone_len)
		
		metrics[ 'BG_hist' ].extend (env.env.env.BG_hist[ :-1 ])
		metrics[ 'CGM_hist' ].extend (env.env.env.CGM_hist[ :-1 ])
		metrics[ 'insulin_hist' ].extend (np.concatenate (env.env.env.insulin_hist).ravel ().tolist ())
		metrics[ 'CHO_hist' ].extend (env.env.env.CHO_hist)
		
		if i_episode % 1000 == 0:
			print (f"Episode: {i_episode + 1}  Reward: {current_return:.3f}")
	
	save_learning_metrics(dir_, **metrics)


if __name__ == '__main__':
	# EXPERIMENTS = "../experiments/"
	EXPERIMENTS = '/home/mila/b/basus/experiments/diabetes/fixed_dose/'
	
	if not os.path.exists (EXPERIMENTS):
		os.makedirs (EXPERIMENTS)
	
	parser = argparse.ArgumentParser ()
	parser.add_argument (
		'--array_id', type=int, default=0, help='(optional) hyperparameter exploration array job id')
	args = parser.parse_args ()
	
	patient_name = 'adult#009'
	reward = 'zone_reward'  # 'zone_reward'
	
	seed = 10
	
	print ('Patient Name:', patient_name, '\n', 'Reward:', reward)
	
	env_id = register_single_patient_env (patient_name,
										  reward_fun=reward,
										  seed=seed,
										  version='-v0')
	env = gym.make (env_id)
	env = DiscretizeActionWrapper (env, low=0, high=5, n_bins=6)
	
	hps = [
		{'hyperglycemic_BG' : 150,
		 'hypoglycemic_BG' : 100,
		 'M_episodes' : 10000,
		 'seed'	: run + 1
		 }
	for run in range(5)
	]
	
	this_runs_hps = hps[args.array_id]
	
	# Create experiment dir
	expt_id = datetime.now ().strftime ('%Y-%m-%d %H:%M:%S') + \
			  '_run' + str (this_runs_hps['seed'])
	
	dir_ = EXPERIMENTS + expt_id
	os.makedirs (dir_)
	print (f"Created {dir_}")
	
	start_time = time.time ()
	fixed_dose(env, dir_, **this_runs_hps)
	end_time = time.time ()
	print ('Total Time:', (end_time - start_time) / 60, 'mins for', this_runs_hps[ 'M_episodes' ], 'episodes.')
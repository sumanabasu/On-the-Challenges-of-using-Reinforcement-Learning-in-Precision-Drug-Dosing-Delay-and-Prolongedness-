import argparse
import gym
import os
import json
import time
from datetime import datetime
from paepomdp.diabetes.trainers.dqn_diabetes_trainer import train
from paepomdp.diabetes.helpers.utils import register_single_patient_env, DiscretizeActionWrapper


if __name__ == '__main__':
	EXPERIMENTS = '../../../Experiments/DQN/'
	
	if not os.path.exists (EXPERIMENTS):
		os.makedirs (EXPERIMENTS)
	
	parser = argparse.ArgumentParser ()
	parser.add_argument (
		'--array_id', type=int, default=0, help='(optional) hyperparameter exploration array job id')
	args = parser.parse_args ()
	
	patient_name = 'adult#009'
	reward = 'zone_reward' #'zone_reward'
	seed = 10
	
	print('Patient Name:', patient_name, '\n', 'Reward:', reward)
	
	env_id = register_single_patient_env (patient_name,
										  reward_fun=reward,
										  seed=seed,
										  version='-v0')
	env = gym.make (env_id)
	env = DiscretizeActionWrapper (env, low=0, high=5, n_bins=6)
	
	# # hyperparameter search
	all_hps = [
		{
			# Fixed
			'state_size': env.observation_space.shape[ 0 ],
			'n_actions': env.num_actions,
			'M_episodes': 10000,
			'replay_buffer_size': 100000,
			'batch_size': 512,
			'eps_start': 0.9,
			'eps_end': 0.05,
			'gamma': 0.999,
			'EXPLORE': 1000,
			'TARGET_UPDATE': 1,
			'hyperglycemic_BG': 150,
			'hypoglycemic_BG': 100,
			
			# Variable
			'learning_rate': lr,
			'eps_decay': decay,
			'state_embedding_size': semb,
			'n_hidden': n_hid,
			'seed': run + 1
		}
		for run in range (5)
		# for lr in [0.0000001]
		for lr in [0.000001]
		for semb in [ 16 ]
		for n_hid in [ 256]
		for decay in [500]
		# for run in range (5)
		# for lr in [0.001, 0.0001, 0.00001, 0.000001]
		# for decay in [ 100, 1000]
	]
	
	this_runs_hps = all_hps[ args.array_id ]
	
	print ('hyper params:', this_runs_hps)
	
	# Create experiment dir
	expt_id = datetime.now ().strftime ('%Y-%m-%d %H:%M:%S') + \
			  '_run' + str (this_runs_hps[ 'seed' ]) + \
			  '_lr' + str (this_runs_hps[ 'learning_rate' ]) + \
			  '_semb' + str (this_runs_hps[ 'state_embedding_size' ]) + \
			  '_nhid' + str (this_runs_hps[ 'n_hidden' ]) + \
			  '_decay' + str (this_runs_hps[ 'eps_decay' ])
	
	dir_ = EXPERIMENTS + expt_id
	os.makedirs (dir_)
	print (f"Created {dir_}")
	
	# save args
	with open (os.path.join (dir_, 'args.json'), 'w') as fp:
		json.dump (this_runs_hps, fp)
	
	start_time = time.time ()
	train (env, dir_, **this_runs_hps)
	end_time = time.time ()
	print ('Total Time:', (end_time - start_time) / 60, 'mins for', this_runs_hps[ 'M_episodes' ], 'episodes.')
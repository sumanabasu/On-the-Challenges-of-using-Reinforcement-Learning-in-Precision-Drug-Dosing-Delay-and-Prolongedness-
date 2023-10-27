import argparse
import gym
import numpy as np
import json
import pickle
import os
import time

from code.algos import utils
from code.algos.q_learning import q_learning

PATH = '../experiments/tabular_q/'

def tune(env,
		 eps_start,
		 eps_end,
		 eps_decay,
		 learning_rate,
		 discount_factor,
		 num_episodes,
		 num_runs,
		 seed,
		 init_pos
		 ):
	if not os.path.exists (PATH):
		os.makedirs (PATH)
	
	runs_x_returns = np.zeros ((num_runs, num_episodes))
	runs_x_lengths = np.zeros ((num_runs, num_episodes))

	for run in range (num_runs):
		
		print ('\n lr:', learning_rate, 'decay:', eps_decay, 'run:', run)
		Q, start_states, runs_x_returns[run], runs_x_lengths[run] = q_learning(env,
																 seed, init_pos,
																 num_episodes,
																 eps_start, eps_end, eps_decay,
																 discount_factor,
																 learning_rate)
		
		seed += 10
	hp_setting = 'lr_' + str (learning_rate) + 'decay_' + str (eps_decay) + '.pkl'
	utils.savePickle (PATH, hp_setting, list ([runs_x_returns, runs_x_lengths]))
	utils.savePickle (PATH, 'Q_' + hp_setting, dict(Q))
	utils.savePickle(PATH, 'SS_' + hp_setting, start_states)
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser ()
	parser.add_argument (
		'--array_id', type=int, default=0, help='(optional) hyperparameter exploration array job id')
	args = parser.parse_args ()
	
	env = gym.make('MoveBlockDiscreteEnv-v0')
	print ('goal:', env.goal_position)
	env = utils.PostionWrapper(env)
	env = utils.DiscretizePositionWrapper(env)
	
	config = [
		{
				'eps_start': 0.9,
				'eps_end': 0.05,
				'eps_decay'	: decay,
	            'learning_rate' : lr,
	            'discount_factor': 0.999,
	            'num_episodes'   :  100000,
	            'num_runs'       : 10,
				'seed'			: 10,
				'init_pos': 'random'
	}
		for lr in [0.005, 0.05, 0.5, 0.99]
		for decay in [500, 10000, 20000, 30000]
	]
	
	this_runs_hps = config[args.array_id]
	print(this_runs_hps)
	start_time = time.time()
	tune(env, **this_runs_hps)
	end_time = time.time()
	print ('Total Time:', (end_time - start_time) / 60, 'mins')
	print(f'Average time per episode per run:', ((end_time - start_time)/this_runs_hps[
		'num_episodes'])/this_runs_hps['num_runs'])
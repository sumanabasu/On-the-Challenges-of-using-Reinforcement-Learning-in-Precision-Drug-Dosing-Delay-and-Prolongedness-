import gym
import numpy as np
import sys
from gym.envs.registration import register
from paepomdp.diabetes.helpers.rewards import zone_reward

def register_single_patient_env (patient_name=None, reward_fun=None, seed=42, version='-v0'):
	"""
	:param patient_name: from the simglucose patient pramas (Eg. adolescent#001) type: str
	:param reward_fun: one of the available reward functions (eg. zone_reward) type: str
	:param seed: environment seed (int)
	:return: simglucose gym patient environment
	"""
	reward_fun = getattr (sys.modules[ __name__ ], reward_fun)
	
	if patient_name is None:
		# Select a random patient
		patients_category = np.random.choice ([ 'adolescent', 'adult', 'child' ])
		patient_id = np.random.randint (1, 10, size=1)
		patient_name = patients_category + '#' + str (patient_id[ 0 ]).zfill (3)
		
		# Create an environment for the selected patient
		env_id = patients_category + str (patient_id[ 0 ]).zfill (3)
	else:
		splits = patient_name.split ('#')
		env_id = splits[ 0 ] + splits[ 1 ]
	
	register (
		id='simglucose-' + env_id + version,
		entry_point='simglucose.envs:T1DSimEnv',
		kwargs={'patient_name': patient_name,
				'reward_fun': reward_fun,
				'seed': seed}
	)
	
	return 'simglucose-' + env_id + version


class DiscretizeActionWrapper(gym.ActionWrapper):
	def __init__ ( self, env, low=0, high=30, n_bins=31):
		super (DiscretizeActionWrapper, self).__init__ (env)
		# _, bins = np.histogram([low, high], bins=n_bins)
		self.min_action = low
		self.max_action = high
		self.bins = np.linspace (low, high, n_bins)
		self.num_actions = np.shape(self.bins)[0]
	
	def _action ( self, continuous_action ):
		discrete_action = self.bins[ np.digitize (continuous_action, self.bins) - 1 ]
		return discrete_action
	
	
def save_learning_metrics(expt_folder, **kwargs):
	for key, value in kwargs.items ():
		np.save (f"{expt_folder}/{key}", value)


if __name__ == '__main__':
	patient_name = 'adult#009'
	reward = 'zone_reward'
	seed = 10
	
	env_id = register_single_patient_env (patient_name,
										  reward_fun=reward,
										  seed=seed,
										  version='-v0')
	env = gym.make(env_id)
	env = DiscretizeActionWrapper(env)
	obs = env.reset()
	action = env.action_space.sample()
	next_obs, reward, done, _ = env.step(action)

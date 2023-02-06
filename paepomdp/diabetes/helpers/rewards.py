import math
import numpy as np

def zone_reward ( BG_last_hour, action):
	target_BG = 125
	error_margin = 25
	max_BG, min_BG = 200, 70
	reward, done = 0, False
	
	bg = BG_last_hour[ -1 ]
	prev_bg = BG_last_hour[-2]
	
	# 1. Hyper and hypoglycemic death penalty (death is bad). End of episode.
	if bg > max_BG or bg <= min_BG:
		reward = -100
		done = True
	# 2. Penalize Hypoglycemia
	if (bg < target_BG - error_margin) and (bg - prev_bg < 0.5):
		reward = -1
	# 3. Penalize Hyperglycemia
	if (bg > target_BG + error_margin) and (bg - prev_bg > 0.5):
		reward = -1
	# 4. Target Zone : Spend as much time as possible in the target BG zone
	elif target_BG + error_margin >= bg >= target_BG - error_margin:
		reward = 10
		
	penalty = math.pow(action, 2) * 0.1
	
	reward -= penalty
	
	return reward, done

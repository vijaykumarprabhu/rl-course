import gym
import random
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
#%matplotlib inline

matplotlib.style.use('ggplot')


def get_epsilon(N_state_count, N_zero=100):
	return N_zero / (N_zero + N_state_count)


def get_action(Q, state, state_count, action_size):
	random_action = random.randint(0, action_size - 1)
	best_action = np.argmax(Q[state])
	epsilon = get_epsilon(state_count)
	return np.random.choice([best_action, random_action], p=[1. - epsilon, epsilon])



def main():
	# This example shows how to perform a single run with the policy that hits for player_sum >= 20
	env = gym.make('Blackjack-v0')
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	returns = defaultdict(lambda: np.zeros(env.action_space.n))
	state_count = defaultdict(float)
	state_action_count = defaultdict(float)

	V = defaultdict(float)

	for i in range(500000):
		episode = list()
		state = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
		done = False
		action = None
		
		#state = choose state how ? 
		#action = np.random.randint(2)

		#ew_state, reward, done, _ = env.step(action)
		#episode.append((state, action, reward))
		#state_count[state] += 1
		#state = new_state
		

		while not done:
			state_count[state] += 1
			# if state[0] >= 20:
			# 	print("stick")
			# 	action = 0            
			# else:
			# 	print("hit")
			# 	action = 1
			action = get_action(Q, state, state_count[state], env.action_space.n)
			new_state, reward, done, _ = env.step(action)
			episode.append((state, action, reward))
			state = new_state
		# finished current episode
		G = 0
		for s, a, r in reversed(episode):
			new_s_a_count = state_action_count[(s, a)] + 1
			
			# for incremental averaging
			G = r + G
			state_action_count[(s, a)] = new_s_a_count
			Q[s][a] = Q[s][a] + (G - Q[s][a]) / new_s_a_count

	print("state value function", Q)
	print("")
	print("")
	## printing tables
	action_dict = {0:"stick", 1:"hit"}
	without_usable_ace = list()
	usable_ace = list()
	for state, val_action in Q.items():
		action = np.argmax(val_action)
		if state[2]==False:
			without_usable_ace.append((state[0], state[1], action_dict[action]))
		else:
			usable_ace.append((state[0], state[1], action_dict[action]))

	print("opt policy without usable ace", without_usable_ace)
	print("")
	print("")
	print("opt policy usable ace", usable_ace)


if __name__ == "__main__":
    main()

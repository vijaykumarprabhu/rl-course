import gym
import numpy as np
import matplotlib.pyplot as plt


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    iterations = 0
    policy = np.zeros(n_states, dtype=np.int) 
    while True:
    	delta = 0.0
    	iterations += 1
    	for state in range(n_states):
    		v = V_states[state]
    		max_action_val = -9999
    		for action in range(n_actions):
    			summation = 0.0
    			for p, n_state, r, is_terminal  in env.P[state][action]:
    				summation += p * (r + gamma* V_states[n_state])
    			if summation > max_action_val:
    				max_action_val = summation
    				policy[state] = action
    		V_states[state] = max_action_val
    		
    		delta = max(delta, abs(v-V_states[state]))
    	if delta < theta:
    		break
    print("steps to converge", iterations)
    print("optimal value function",V_states)

    ## computing optimal policy
    return V_states



def choose_abs_greedy_action(state, Q, epsilon):
	action = None
	if np.random.uniform(0, 1) < epsilon:
		action = np.random.randint(env.action_space.n)
	else:
		result = np.where(Q[state,:] == np.amax(Q[state,:]))
		#m = max(Q[state,:])
		#max_indices = [i for i, j in enumerate(Q[state,:]) if j == m]
		action = np.random.choice(result[0])
	return action


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n,  env.action_space.n))
    # TODO: implement the sarsa algorithm

    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        s = env.reset()
        done = False
        t = 0 
        T = np.inf
        a = choose_abs_greedy_action(s, Q, epsilon)

        actions = [a]
        states = [s]
        rewards = [0]

        while True:
        	if t < T:
       			s_, r, done, _ = env.step(a)
        		states.append(s_)
        		rewards.append(r)
        		if done:
        			T = t + 1
        		else:
        			a = choose_abs_greedy_action(s_, Q, epsilon)
        			actions.append(a)
        
        	# tau -which timestamp to update if t=5 than tau=2 nd time stamp to be updated
        	tau = t - n + 1
        	if tau >= 0:
        		G = 0
        		for i in range(tau + 1, min(tau + n + 1, T + 1)):
        			G += np.power(gamma, i - tau - 1) * rewards[i]

        		if tau + n < T:
        			state_action = (states[tau + n], actions[tau + n])
        			G += np.power(gamma, n) * Q[state_action[0]][state_action[1]]

        		state_action = (states[tau], actions[tau])
        		Q[state_action[0]][state_action[1]] += alpha * (G - Q[state_action[0]][state_action[1]])

        	if tau == T - 1:
        		break
        	t += 1
    
    return Q

    pass


env=gym.make('FrozenLake-v0', map_name="8x8")
n_states = env.observation_space.n
n_actions = env.action_space.n

# getting actual state values from dp

actual_state_values = value_iteration()

#print(actual_state_values)

# TODO: run multiple times, evaluate the performance for different n and alpha
#Q = nstep_sarsa(env)
#print("####")
#print(Q)

alpha_range = np.linspace(0, 1, 6)
n_range = np.power(2, range(10))

sq_errors = {}

for n in n_range:
	ers = []
	for alpha in alpha_range:
		print("running estimation for alpha={} and n={}".format(alpha, n))
		current_Q = nstep_sarsa(env, n=n, alpha=alpha)
		print("*****")
		#print(current_Q)
		#estimate_state_values = [np.mean(list(v.values())) for v in current_Q.values()]
		estimate_state_values = [np.mean(v) for v in current_Q]
		ers.append(np.mean([er ** 2 for er in actual_state_values - np.array(estimate_state_values)]))
	sq_errors[n] = ers

plt.figure(figsize=[10, 6])
for n in n_range:
	plt.plot(alpha_range, sq_errors[n], label="n={}".format(n))
plt.xlabel('learning rate')
plt.ylabel('RMS error')
plt.legend()
plt.show()

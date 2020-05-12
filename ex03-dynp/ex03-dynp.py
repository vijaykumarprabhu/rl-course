import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# custom_map3x3 = [
#     'SFF',
#     'FFF',
#     'FHG',
# ]
# env = gym.make("FrozenLake-v0", desc=custom_map3x3)

# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


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
    return policy

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()

import gym
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
#%matplotlib inline

matplotlib.style.use('ggplot')



def plot_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    surf = ax.plot_surface(X, Y, Z, vmin=-1.0, vmax=1.0)

    ax.set_xlabel('Player sum')
    ax.set_ylabel('Dealer showing')
    ax.set_zlabel('Value')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.view_init(ax.elev, 120)
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V):
    
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_noace = Z_noace.reshape((Z_noace.shape[0],Z_noace.shape[1]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    Z_ace = Z_ace.reshape((Z_ace.shape[0], Z_ace.shape[1]))
    
    print("Xshape",X.shape)
    print("Yshape",X.shape)
    print("Zshape",Z_noace.shape)
    
    plot_surface(X, Y, Z_noace, "no useable_ace")
    plot_surface(X, Y, Z_ace, "usable ace")




def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    #returns = defaultdict(lambda: np.zeros(env.action_space.n))
    V = defaultdict(float)
    returns = defaultdict(lambda: np.zeros(1))
    obs_count = defaultdict(float)


    for i in range(500000):
        episode = list()
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        while not done:
            obs_count[obs] += 1.0
            #print("observation:", obs)
            if obs[0] >= 20:
                #print("stick")
                obs, reward, done, _ = env.step(0)
            else:
                #print("hit")
                obs, reward, done, _ = env.step(1)
            returns[obs] += reward
            episode.append((obs,reward))
            #print("reward:", reward)
            #print("")

        for state in obs_count.keys():
            V[state] = returns[state]/obs_count[state]

    print("value_function", V)
    plot_value_function(V)

if __name__ == "__main__":
    main()

import gym
import numpy as np
import random as pr
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs.registration import register

def rargmax(vector): # random argmax의 약자
    """
    Argmax that chooses randomly among eligible maximum indices.
    """
    m = np.amax(vector)
    condition = (vector == m)
    indices = np.nonzero(condition)[0]
    return pr.choice(indices)

register(id = 'FrozenLake-v3',
         entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
         kwargs = {'map_name': '4x4',
                   'is_slippery': False})
env = gym.make(id = 'FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
num_eposides = 2000

# create lists to contain total rewards and steps per episode
rList = []

for i in range(num_eposides):
    # Reset environment and get first new observation
    state = env.reset() # [?] start 지점을 의미하는 state?
    # print("state.shape: ", state.shape)
    rAll = 0
    done = False # Hole에 빠지거나, Goal에 도달

    # The Q-Table learning algorithm
    while not done:
        print("state: ", state)
        action = rargmax(Q[state, :]) # [??]
        # test/2.numpy.nonzero.py를 보면,
        # [False False  True False]
        # [LEFT  DOWN   RIGHT   UP]

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: ", str(sum(rList)/num_eposides))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

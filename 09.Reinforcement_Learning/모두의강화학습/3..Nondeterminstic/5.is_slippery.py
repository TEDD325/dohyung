# https://github.com/hunkim/ReinforcementZeroToAll
# http://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user

import gym
import readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {'\x1b[A': UP,
              '\x1b[B': DOWN,
              '\x1b[C': RIGHT,
              '\x1b[D': LEFT}

# register(
#     id='FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4',
#             'is_slippery': False}
# )

# is_slippery True(Default)
env = gym.make('FrozenLake-v0')

env.reset()

# print_utils.clear_screen()
env.render()  # Show the initial board

while True:
    # Choose an action from keyboard
    key = readchar.readkey()

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)

    # Show the board after action
    # print_utils.clear_screen()
    env.render()

    print("State: {} Action: {} Reward: {} Info: {}".format(
        state, action, reward, info))

    if done:
        print("GAME OVER")
        print("Finished with reward: ", reward)
        break
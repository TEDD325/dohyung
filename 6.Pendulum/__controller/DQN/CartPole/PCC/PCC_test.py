from pylab import *
import numpy as np
from environment_1 import Env as Env_1
from environment_2 import Env as Env_2
import pickle

EPISODES = 1000
HISTORY_LENGTH = 8

def manual_balance(filename, env_num):
    if env_num == 1:
        env = Env_1()
    else:
        env = Env_2()

    state_size = env.state_space_shape[0]
    episode_start_num = 0

    for episode in range(episode_start_num, EPISODES):
        pickle_data = []
        done = False
        step = 0

        state, theta_n_k1, theta_dot_k1, alpha_n_k1, alpha_dot_k1 = env.reset()
        state_for_manual_balance = state
        state = [state[0] * 100 , state[1], state[2] * 100, state[3]]
        state = np.reshape(state, [1, state_size, 1, 1])
        history = np.zeros([1, state_size, HISTORY_LENGTH, 1])
        for i in range(HISTORY_LENGTH):
            history = np.delete(history, 0, axis=2)
            history = np.append(history, state, axis=2)

        while not done:
            step += 1

            kp_theta = 2.0
            kd_theta = -2.0
            kp_alpha = -30.0
            kd_alpha = 2.5

            alpha = state_for_manual_balance[0]
            theta = state_for_manual_balance[2]

            theta_n = -theta
            theta_dot = (50.0 * theta_n) - (50.0 * theta_n_k1) + (0.7612 * theta_dot_k1)  # 5ms
            theta_n_k1 = theta_n
            theta_dot_k1 = theta_dot

            alpha_n = -alpha
            alpha_dot = (50.0 * alpha_n) - (50.0 * alpha_n_k1) + (0.7612 * alpha_dot_k1)  # 5ms
            alpha_n_k1 = alpha_n
            alpha_dot_k1 = alpha_dot

            motor_voltage = (theta * kp_theta) + (theta_dot * kd_theta) + (alpha * kp_alpha) + (
                        alpha_dot * kd_alpha)

            if motor_voltage > 15.0:
                motor_voltage = 15.0
            elif motor_voltage < -15.0:
                motor_voltage = -15.0

            motor_voltage = -motor_voltage

            motorPWM = motor_voltage * (625.0 / 15.0)
            motorPWM = int(motorPWM)
            motorPWM += -(motorPWM % 50)

            if motorPWM < 0:
                action_index = 0
            elif 0 < motorPWM:
                action_index = 2
            else:
                action_index = 1

            pickle_data.append(history[:,:,-1,:])

            next_state, reward, done, info = env.step(action_index)

            state_for_manual_balance = next_state
            next_state = [next_state[0] * 100, next_state[1], next_state[2] * 100, next_state[3]]
            next_state = np.reshape(next_state, (1, state_size, 1, 1))
            history = np.delete(history, 0, axis=2)
            history = np.append(history, values=next_state, axis=2)
            history = np.reshape(history, [1, state_size, HISTORY_LENGTH, 1])

        with open(filename, 'wb') as f:
            pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

        env.wait()

env_num = 2
# filename = 'pcc_pendulum_1-1.pickle'
# filename = 'pcc_pendulum_1-2.pickle'
# filename = 'pcc_pendulum_2-1.pickle'
# filename = 'pcc_pendulum_2-2.pickle'
filename = 'pcc_pendulum_3-1.pickle'
# filename = 'pcc_pendulum_3-2.pickle'

manual_balance(filename, env_num)
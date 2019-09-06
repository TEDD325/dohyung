from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from environment_cnn_dqn import Env
from datetime import datetime
from collections import deque
from matplotlib import pylab
import _pickle as pickle
import numpy as np
from pylab import *
import random
import time
import sys
import os

ACTION_TIME = 0.006 # [MEMO of dohk] 6 ms

EPISODES = 100000
HISTORY_LENGTH = 8
MEMORY_LENGTH = 20000

TRAIN_ITERATION_N = 100

INITIAL_EPSILON = 0.3
EPSILON_DECATOR = 0.98
EPSILON_MINIMUM = 0.001

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001

BATCH_SIZE = 256

SAVED_MODEL_PATH = "./save/rotary_pendulum.h5"
SAVED_MEMORY_PATH = "./save/dqn_memory.pickle"
GRAPH_PATH = "./save/rotary_pendulum.png"
LOSS_GRAPH_PATH = "./save/rotary_pendulum_loss.png"
GRAPH_DATA_PATH = "./save/graph_data.pickle"
LAST_EPISODE_N_PATH = "./save/last_episode_num.txt"


# DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False

        # define state and size of action
        self.state_size = state_size
        self.action_size = action_size

        # self.epsilon = INITIAL_EPSILON

        # replay memory, maximum size is 20000
        self.memory = deque(maxlen=MEMORY_LENGTH)

        self.loss = None

        self.graph_data = list()

        # generate model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if os.path.exists(SAVED_MODEL_PATH):
            self.load_model = True

        if self.load_model:
            self.model.load_weights(SAVED_MODEL_PATH) # [MEMO of dohk] no definition.. 190219
            self.memory_load(SAVED_MEMORY_PATH)

    def graph_data_dump(self, file_name):
        with open(file_name, 'wb') as memory_file:
            pickle.dump(self.graph_data, memory_file)

    def memory_dump(self, file_name):
        with open(file_name, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)

    def memory_load(self, file_name):
        with open(file_name, 'rb') as memory_file:
            self.memory = pickle.load(memory_file)

    # generate Artificial Neural Network whose input is state and output is Q-function
    def build_model(self):
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(4, 4),
                   activation='relu',
                   input_shape=[self.state_size, HISTORY_LENGTH, 1]
            )
        )
        model.add(Conv2D(64, (1, 4), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    # update weight of target model into weight of model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        # [MEMO of dohk] target_model.set_weights; no definition.. 190219
        # [MEMO of dohk] model.get_weights(); no definition.. 190219

    # select action based on epsilon greedy policy
    def get_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        #     q_value = self.model.predict(state)
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # save sample <s, a, r, s'>into replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # learning model using batch extracted from replay memory<=== pub
    def train_model(self):
        # extract sample randomly, whose size is batch size, from memory
        mini_batch = random.sample(self.memory, BATCH_SIZE)

        states = np.zeros(
            (BATCH_SIZE, self.state_size, HISTORY_LENGTH, 1)
        )

        next_states = np.zeros(
            (BATCH_SIZE, self.state_size, HISTORY_LENGTH, 1)
        )

        actions, rewards, dones = [], [], []

        for i in range(BATCH_SIZE):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # model Q-function about current state
        # Q-function of target model about next state
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # update target using bellman optimal equation
        for i in range(BATCH_SIZE):
            if dones[i]: # [MEMO of dohk] terminal state -190220
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + DISCOUNT_FACTOR * (np.amax(target_val[i]))

        hist = self.model.fit(states, target, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        self.loss = np.mean(hist.history['loss'])


if __name__ == "__main__":
    # print(str(datetime.now()) + ' started!', flush=True)
    print(str(datetime.now()) + ' started!')

    env = Env()

    state_size = env.state_space_shape[0]
    action_size = env.action_space_shape[0]

    # Reload episode number of last learning
    episode_start_num = 0
    if os.path.exists(LAST_EPISODE_N_PATH):
        with open(LAST_EPISODE_N_PATH, 'r') as f:
            episode_start_num = int(f.readline())

    # generate DQN agent
    agent = DQNAgent(state_size, action_size)

    scores, episodes, graph_episodes, losses = [], [], [], []
    success_cnt = 0
    e = 0

    for episode in range(episode_start_num, EPISODES):
        # agent.epsilon = INITIAL_EPSILON
        done = False
        score = 0
        step = 0

        # init env
        # previous_time = time.perf_counter()

        # state = [pendulum_radian, pendulum_velocity, motor_radian, motor_velocity]
        state, theta_n_k1, theta_dot_k1, alpha_n_k1, alpha_dot_k1 = env.reset()
        state_for_manual_balance = state
        state = [state[0] * 100 , state[1], state[2] * 100, state[3]] # radian * 100

        state = np.reshape(state, [1, state_size, 1, 1]) # [MEMO of dohk] reference: test/np.reshape.py
        history = np.zeros([1, state_size, HISTORY_LENGTH, 1]) # [MEMO of dohk] reference: test/np.zeros.py
        for i in range(HISTORY_LENGTH):
            history = np.delete(history, 0, axis=2) # [MEMO of dohk] reference: test/np.delete.py
            history = np.append(history, state, axis=2) # [MEMO of dohk] reference: test/np.delete.py
        history = np.reshape(history, [1, state_size, HISTORY_LENGTH, 1]) # [MEMO of dohk] 필요하지 않은 것 같은데.. 한 번 더 reshape을 할 이유가 없다.-190219

        if len(agent.memory) >= MEMORY_LENGTH - 1:
            e += 1

        while not done:
            step += 1

            previous_time = time.perf_counter()

            # balance using network's output
            #if len(agent.memory) >= MEMORY_LENGTH - 1:
            action_index = agent.get_action(state=history)
            # manual balance
            # else:
            #     # if np.random.rand() <= agent.epsilon:   # random action for noise
            #     #     action_index = random.randrange(agent.action_size)
            #     # else:
            #     kp_theta = 2.0
            #     kd_theta = -2.0
            #     kp_alpha = -30.0
            #     kd_alpha = 2.5
            #
            #     # transfer function = 50s/(s+50)
            #     # z-transform at 1ms = (50z - 50)/(z-0.9512)
            #     alpha = state_for_manual_balance[0]
            #     theta = state_for_manual_balance[2]
            #
            #     theta_n = -theta
            #     theta_dot = (50.0 * theta_n) - (50.0 * theta_n_k1) + (0.7612 * theta_dot_k1)  # 5ms
            #     theta_n_k1 = theta_n
            #     theta_dot_k1 = theta_dot
            #
            #     # transfer function = 50s/(s+50)
            #     # z-transform at 1ms = (50z - 50)/(z-0.9512)
            #     alpha_n = -alpha
            #     alpha_dot = (50.0 * alpha_n) - (50.0 * alpha_n_k1) + (0.7612 * alpha_dot_k1)  # 5ms
            #     alpha_n_k1 = alpha_n
            #     alpha_dot_k1 = alpha_dot
            #
            #     # multiply by proportional and derivative gains
            #     motor_voltage = (theta * kp_theta) + (theta_dot * kd_theta) + (alpha * kp_alpha) + (
            #                 alpha_dot * kd_alpha)
            #
            #     # set the saturation limit to +/- 15V
            #     if motor_voltage > 15.0:
            #         motor_voltage = 15.0
            #     elif motor_voltage < -15.0:
            #         motor_voltage = -15.0
            #
            #     # invert for positive CCW
            #     motor_voltage = -motor_voltage
            #
            #     # convert the analog value to the PWM duty cycle that will produce the same average voltage
            #     motorPWM = motor_voltage * (625.0 / 15.0)
            #     motorPWM = int(motorPWM)
            #     motorPWM += -(motorPWM % 50)
            #
            #     if motorPWM < 0:
            #         action_index = 0
            #     elif 0 < motorPWM:
            #         action_index = 2
            #     else:
            #         action_index = 1

            prior_history = history

            # proceed only one timestep from env based on action selected
            next_state, reward, done, info = env.step(action_index)

            state_for_manual_balance = next_state
            next_state = [next_state[0] * 100, next_state[1], next_state[2] * 100, next_state[3]]

            next_state = np.reshape(next_state, (1, state_size, 1, 1))
            history = np.delete(history, 0, axis=2)
            history = np.append(history, values=next_state, axis=2)

            history = np.reshape(history, [1, state_size, HISTORY_LENGTH, 1])
            # save sample <s, a, r, s'> at replay memory
            if not(done and step == 1):
                # agent.append_sample(state, action_index, reward, next_state, done)
                agent.append_sample(prior_history, action_index, reward, history, done)

            score += reward

            # if agent.epsilon > EPSILON_MINIMUM:
            #     agent.epsilon *= EPSILON_DECATOR

            if not done:
                while True:
                    current_time = time.perf_counter()
                    if current_time - previous_time >= 6 / 1000: # 60 ms
                        break
            else:
                env.wait()

                # if len(agent.memory) >= MEMORY_LENGTH and step > 5:
                if len(agent.memory) >= BATCH_SIZE and step > 5:
                    # Train model
                    tn = 0 # [MEMO of dohk] tn: train iteration num
                    for i in range(TRAIN_ITERATION_N):
                        agent.train_model()
                        tn += 1
                    print("train iteration num -", tn)

                    # Update target model
                    agent.update_target_model()

                    # Graph's x: episodes, y: scores
                    if episode > 0:
                        scores.append(score)
                        episodes.append(episode)
                        graph_episodes.append(e)
                        losses.append(agent.loss)

                        fig = plt.figure(1)
                        plt.plot(graph_episodes, scores, 'b')
                        fig.savefig(GRAPH_PATH)
                        fig.clear()
                        plt.close()
                        fig = plt.figure(2)
                        plt.plot(graph_episodes, losses, 'r')
                        fig.savefig(LOSS_GRAPH_PATH)
                        fig.clear()
                        plt.close()

                    # Grpah's data: [episodes, scores, losses]
                    # print('graph data: {0}'.format([score, episode, agent.loss]))
                    agent.graph_data.append([score, episode, agent.loss])
                    agent.graph_data_dump(GRAPH_DATA_PATH)

                    #print("episode:{0}  score:{1}  step:{2}  info:{3}  memory length:{4}  epsilon:{5:10.8f}".format(
                    #    episode, score, step, info, len(agent.memory), agent.epsilon
                    #), flush=True)

                    # print("episode:{0}  score:{1}  step:{2}  info:{3}  memory length:{4}  epsilon:{5:10.8f}".format(
                    #     episode, score, step, info, len(agent.memory), agent.epsilon
                    # ))
                    print("episode:{0}  score:{1}  step:{2}  info:{3}  memory length:{4}".format(
                        episode, score, step, info, len(agent.memory)
                    ))

                    # Save results
                    agent.model.save_weights(SAVED_MODEL_PATH)
                    agent.memory_dump(SAVED_MEMORY_PATH)

                    f = open(LAST_EPISODE_N_PATH, 'w')
                    f.write(str(episode) + "\n")
                    f.close()

                    sys.stdout.flush()

                    # if serial successes are persisted, system will be shutted down.
                    if step >= 5000:
                        success_cnt += 1
                    else:
                        success_cnt = 0

                    if success_cnt >= 5:
                        env.close()
                        sys.exit()

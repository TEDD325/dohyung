import threading

THREAD_NUM = 2
# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

EPISODES = 1000
global_param = {}
shared_memory_0 = deque(maxlen=1)
shared_memory_1 = deque(maxlen=1)


class DQNAgent:
    def __init__(self, state_size, action_size):
        global sess
        self.state_size = state_size
        self.action_size = action_size
        # print(self.state_size, self.action_size)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.__global_score_list_lock = threading.RLock()

        # model = Sequential()
        # layer_1 =
        # layer_2 = layer_1.add(Dense(24, activation='relu', name='Dense_layer_2'))
        # self.model = self._build_model(flag, model)
        with self.__global_score_list_lock:
            self.model = self._build_model()

        self.layer_output = [layer.output for layer in self.model.layers]

    def _build_model(self):
        # input = Input(shape=(4,))
        # model = Sequential([
        #     Dense(24, input_dim=self.state_size, activation='relu', name='Dense_layer_1'),
        #     Dense(24, activation='relu', name='Dense_layer_2'),
        #     Dense(self.action_size, activation='linear', name='Dense_last_layer')
        # ])
        #
        #
        # layer_1 = model.layers[0](input)
        # layer_2 = model.layers[1](layer_1)
        # layer_3 = model.layers[2](layer_2)
        # # layer_3.compile(loss='mse',
        # #               optimizer=Adam(lr=self.learning_rate))
        #
        # output = Model(inputs=input, outputs=layer_3)
        # return output

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', name='Dense_layer_1'))
        model.add(Dense(24, activation='relu', name='Dense_layer_2'))
        model.add(Dense(self.action_size, activation='linear', name='Dense_last_layer'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            with self.__global_score_list_lock:
                hist = self.model.fit(state, target_f, epochs=1, verbose=0)
            self.loss = np.mean(hist.history['loss'])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.loss

    def load(self, name):
        with self.__global_score_list_lock:
            self.model.load_weights(name)

    def save(self, name):
        with self.__global_score_list_lock:
            self.model.save_weights(name)

class DQN:
    def __init__(self):
        # config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        # sess = tf.Session(config=config)
        # K.set_session(sess)  # K is keras backend
        pass

    def train(self):
        Solver = [DQN_solver(idx) for idx in range(THREAD_NUM)]
        for dqn_agent in Solver:
            dqn_agent.start()

class DQN_solver(threading.Thread):
    def __init__(self, idx, ):
        threading.Thread.__init__(self)
        self.thread_id = idx
        self.loss = 0.0
        self.__global_score_list_lock = threading.RLock()

        # self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        # self.sess = tf.Session(config=self.config)
        # K.set_session(self.sess)  # K is keras backend

    def run(self):
        # with self.sess.graph.as_default():
        with self.__global_score_list_lock:
            env = gym.make('CartPole-v1')
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = DQNAgent(state_size, action_size)
            # print("agent:",agent)
            # agent.load("./save/cartpole-dqn.h5")
            done = False
            batch_size = 32

            for e in range(EPISODES):
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                for time in range(500):
                    # env.render()
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    reward = reward if not done else -10
                    next_state = np.reshape(next_state, [1, state_size])
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    # print("agent.layer_output[0]: ", agent.model.layers[0].get_weights()[0])
                    # print("agent.layer_output[1]: ", agent.model.layers[1].get_weights()[0])
                    # print("shape of agent.layer_output[0]: ", len(agent.model.layers[0].get_weights()))

                    # print("agent.layer_output[0]: ", agent.layer_output[0])
                    # print("agent.layer_output[1]: ", agent.layer_output[1])

                    # print(id(agent.model.layers[0]))
                    # print(agent.layer_output[0])
                    if done and len(agent.memory) > batch_size:
                        self.loss = agent.replay(batch_size)
                        shared_memory_0.append([0,
                                                0.0,
                                                None,
                                                None])
                        shared_memory_1.append([0,
                                                0.0,
                                                None,
                                                None])
                        # print("[INFO] agent.model.layers[0].get_weights():", agent.model.layers[0].get_weights())
                        # print("22")
                        # print("[INFO] agent.model.layers[0].get_weights():", agent.model.layers[0].get_weights()[0])
                        # print("2222")
                        # print("[INFO] agent.model.layers[0].get_weights():", agent.model.layers[0].get_weights()[0][0])
                        # print("222222")
                        if self.thread_id == 0:
                            shared_memory_0.append([self.thread_id,
                                                  self.loss,
                                                  agent.model.layers[0].get_weights(),
                                                  agent.model.layers[1].get_weights()])
                            # print("shared_memory_0[0]: ", shared_memory_0[0])
                            # print("shared_memory_0[0][1]: ", shared_memory_0[0][1])
                            # print("shared_memory_0[0][2]: ", shared_memory_0[0][2])
                            # print("shared_memory_0[0][3]: ", shared_memory_0[0][3])

                            if self.loss > shared_memory_1[0][1]:
                                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                print("0-", self.loss, shared_memory_1[0][1])

                                # agent.model.layers[0].set_weights(shared_memory_1[0][2])
                                # agent.model.layers[1].set_weights(shared_memory_1[0][3])

                        elif self.thread_id == 1:
                            shared_memory_1.append([self.thread_id,
                                                    self.loss,
                                                    agent.model.layers[0].get_weights(),
                                                    agent.model.layers[1].get_weights()])
                            # print("shared_memory_1[0][1]: ", shared_memory_1[0][1])
                            # print("shared_memory_1[0][2]: ", shared_memory_1[0][2])
                            # print("shared_memory_1[0][3]: ", shared_memory_1[0][3])
                            if self.loss > shared_memory_0[0][1]:
                                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                print("1-", self.loss, shared_memory_0[0][1])
                                # agent.model.layers[0].set_weights(shared_memory_0[0][2])
                                # agent.model.layers[1].set_weights(shared_memory_0[0][3])

                            # print("shared_memory_1: ", shared_memory_1)

                        # print(self.thread_id, "th thread -", "episode: {}/{}, score: {}, e: {:.2}, loss: {:.5}"
                        #       .format(e, EPISODES, time, agent.epsilon, self.loss))


                        break
                    if done and not len(agent.memory) > batch_size:
                        pass
                        # print(self.thread_id, "th thread -", "episode: {}/{}, score: {}, e: {:.2}"
                        #       .format(e, EPISODES, time, agent.epsilon))

                # if e % 10 == 0:
                #     agent.save("./save/cartpole-dqn.h5")

if __name__ == "__main__":
    dqn = DQN()
    dqn.train()
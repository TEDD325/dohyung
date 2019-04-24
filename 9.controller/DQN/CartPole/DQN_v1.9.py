import threading
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from score_logger import ScoreLogger
import os
import tensorflow as tf
from keras import backend as K

SCORE_PATH = "./scores/"

if not os.path.exists(SCORE_PATH):
    os.makedirs(SCORE_PATH)

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

THREAD_NUM = 2
global_loss = 1e+10
global_Network = None


class DQN:
    def __init__(self):
        pass

    def train(self):
        Solver = [DQN_Agent(idx) for idx in range(THREAD_NUM)]
        for dqn_agent in Solver:
            dqn_agent.start()

class DQN_Agent(threading.Thread):
    def __init__(self, idx):
        threading.Thread.__init__(self)
        self.thread_id = idx
        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.state_size = None
        self.action_size = None
        self.loss = 1e+10
        self.Network = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, model):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, state_next, terminal in batch:
            q_update = reward

            if not terminal:
                q_update = (reward + GAMMA * np.amax(model.predict(state_next)[0]))

            q_values = model.predict(state)
            q_values[0][action] = q_update

            hist = model.fit(state, q_values, epochs=1, verbose=0)


        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        return hist

    def act(self, state, model):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_size)

        q_values = model.predict(state)

        return np.argmax(q_values[0])

    def model_Thread_0(self):
        model = Sequential()
        model.add(Dense(24,
                         input_shape=(self.state_size,),
                         activation="relu"))
        model.add(Dense(24,
                         activation="relu"))
        model.add(Dense(self.action_size,
                         activation="linear"))
        model.compile(loss="mse",
                       optimizer=Adam(lr=LEARNING_RATE))

        return model

    def model_Thread_1(self):
        model = Sequential()
        model.add(Dense(24,
                        input_shape=(self.state_size,),
                        activation="relu"))
        model.add(Dense(24,
                        activation="relu"))
        model.add(Dense(self.action_size,
                        activation="linear"))
        model.compile(loss="mse",
                      optimizer=Adam(lr=LEARNING_RATE))

        return model

    def train(self, model, env, state, terminal, step, run):
        pass

    def run(self):
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        sess = tf.Session(config=config)
        K.set_session(sess)

        with sess.graph.as_default():

            global global_loss
            global global_Network
            # CartPole-v1 환경
            env = gym.make(ENV_NAME)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

            score_logger = ScoreLogger(ENV_NAME)

            self.state_size = state_size
            self.action_size = action_size

            # 모델 생성
            self.model_0 = self.model_Thread_0()
            self.model_1 = self.model_Thread_1()


            run = 0
            while True:
                run += 1
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                step = 0

                while True:
                    if self.thread_id == 0:
                        step += 1
                        # env.render()
                        action = self.act(state, self.model_0)
                        state_next, reward, terminal, info = env.step(action)

                        reward = reward if not terminal else -reward

                        state_next = np.reshape(state_next, [1, state_size])

                        self.remember(state, action, reward, state_next, terminal)
                        state = state_next

                        if terminal:
                            score_logger.add_score(step, run)
                            hist = self.experience_replay(self.model_0)
                            if type(hist) != type(None):
                                print("thread id:", self.thread_id,
                                      "Run: " + str(run) + " loss: " + str(hist.history['loss'][0]), "score: " + str(
                                        step))

                                # sess = tf.Session(graph=self.model_0.output.graph)
                                # sess.run(tf.global_variables_initializer())

                                self.loss = hist.history['loss'][0]

                                # self.Network = [[self.model_0.layers[0].weights[0].eval(session=sess),
                                #                  self.model_0.layers[0].weights[1].eval(session=sess)],
                                #                 [self.model_0.layers[1].weights[0].eval(session=sess),
                                #                  self.model_0.layers[1].weights[1].eval(session=sess)]]


                                self.Network = [self.model_0.layers[0].get_weights(), self.model_0.layers[1].get_weights()]

                                if self.loss < global_loss:
                                    print("global_loss:", global_loss, "self.loss:", self.loss)
                                    print("gloabl_Network:", global_Network)
                                    print()

                                    global_loss = self.loss
                                    global_Network = self.Network
                                    update_flag = True

                                    print("global_loss:", global_loss, "self.loss:", self.loss)
                                    print("gloabl_Network:", global_Network)
                                    print()

                                elif self.loss > global_loss + 1e-7 and update_flag:
                                    print("Entering elif statement")
                                    self.model_0.layers[0].set_weights(global_Network[0])
                                    self.model_0.layers[1].set_weights(global_Network[1])

                                    update_flag = False

                            break
                    elif self.thread_id == 1:
                        step += 1
                        # env.render()
                        action = self.act(state, self.model_1)
                        state_next, reward, terminal, info = env.step(action)

                        reward = reward if not terminal else -reward

                        state_next = np.reshape(state_next, [1, state_size])

                        self.remember(state, action, reward, state_next, terminal)
                        state = state_next

                        if terminal:
                            score_logger.add_score(step, run)
                            hist = self.experience_replay(self.model_1)
                            if type(hist) != type(None):
                                print("thread id:", self.thread_id,
                                      "Run: " + str(run) + " loss: " + str(hist.history['loss'][0]), "score: " + str(
                                        step))

                                sess = tf.Session(graph=self.model_1.output.graph)
                                sess.run(tf.global_variables_initializer())

                                self.loss = hist.history['loss'][0]

                                self.Network = [[self.model_1.layers[0].weights[0].eval(session=sess),
                                                 self.model_1.layers[0].weights[1].eval(session=sess)],
                                                [self.model_1.layers[1].weights[0].eval(session=sess),
                                                 self.model_1.layers[1].weights[1].eval(session=sess)]]

                                if self.loss < global_loss:
                                    print("global_loss:", global_loss, "self.loss:", self.loss)
                                    print("gloabl_Network:", global_Network)
                                    print()

                                    global_loss = self.loss
                                    global_Network = self.Network
                                    update_flag = True

                                    print("global_loss:", global_loss, "self.loss:", self.loss)
                                    print("gloabl_Network:", global_Network)
                                    print()

                                elif self.loss > global_loss + 1e-7 and update_flag:
                                    print("Entering elif statement")
                                    self.model_1.layers[0].set_weights(global_Network[0])
                                    self.model_1.layers[1].set_weights(global_Network[1])

                                    update_flag = False
                            break



if __name__ == "__main__":
    dqn = DQN()
    dqn.train()
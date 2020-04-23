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
loss = 1e+10
Network = None


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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, state_next, terminal in batch:
            q_update = reward

            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))

            q_values = self.model.predict(state)
            q_values[0][action] = q_update

            hist = self.model.fit(state, q_values, epochs=1, verbose=0)


        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        return hist

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state)

        return np.argmax(q_values[0])

    def run(self):
        # CartPole-v1 환경
        env = gym.make(ENV_NAME)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        self.state_size = state_size
        self.action_size = action_size

        # 모델 생성

        self.model = Sequential()
        self.model.add(Dense(24,
                             input_shape=(self.state_size,),
                             activation="relu"))
        self.model.add(Dense(24,
                             activation="relu"))
        self.model.add(Dense(self.action_size,
                             activation="linear"))
        self.model.compile(loss="mse",
                           optimizer=Adam(lr=LEARNING_RATE))

        score_logger = ScoreLogger(ENV_NAME)


        run = 0
        while True:
            run += 1
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            step = 0

            while True:
                step += 1
                # env.render()
                action = self.act(state)
                state_next, reward, terminal, info = env.step(action)

                reward = reward if not terminal else -reward

                state_next = np.reshape(state_next, [1, state_size])

                self.remember(state, action, reward, state_next, terminal)
                state = state_next

                if terminal:
                    score_logger.add_score(step, run)
                    hist = self.experience_replay()
                    if type(hist) != type(None):
                        print("thread id:", self.thread_id, "Run: " + str(run) + " loss: " + str(hist.history['loss'][0]), "score: " + str(
                            step))
                        with self.graph.as_default():
                            loss = hist.history['loss'][0]
                            Network = self.model.layers[0]
                        sess = tf.Session(graph=self.model.output.graph)
                        sess.run(tf.global_variables_initializer())
                        print('loss-',loss,'Network-',Network.weights[0].eval(session=sess))
                    break



if __name__ == "__main__":
    dqn = DQN()
    dqn.train()
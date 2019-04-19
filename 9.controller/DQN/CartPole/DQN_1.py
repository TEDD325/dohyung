import threading
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from score_logger import ScoreLogger
import os

ENV_NAME = "CartPole-v1"
SCORE_PATH = "./scores/"
MODEL_PATH = "./save_model/"
GRAPTH_PATH = "./save_graph/"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(GRAPTH_PATH):
    os.makedirs(GRAPTH_PATH)

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

THREAD_NUM = 2

global_loss = None
global_network = None

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.state_size = observation_space
        self.action_size = action_space
        self.learning_rate = 0.001
        self.memory = deque(maxlen=MEMORY_SIZE)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='elu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='elu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

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
            # print("state: ", state)
            # print("q_value: ", q_values)
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

class DQN:
    def __init__(self):
        if not os.path.exists(SCORE_PATH):
            os.makedirs(SCORE_PATH)

    def train(self):
        Solver = [DQN_solver(thread_id) for thread_id in range(THREAD_NUM)]
        for dqn_agent in Solver:
            dqn_agent.start()

class DQN_solver(threading.Thread):
    def __init__(self, thread_id):
        threading.Thread.__init__(self)
        self.thread_id = thread_id

    def run(self):
        env = gym.make(ENV_NAME)
        score_logger = ScoreLogger(ENV_NAME)
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        dqn_solver = DQNSolver(observation_space, action_space)
        run = 0
        while True:
            run += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:
                step += 1
                # env.render()
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, observation_space])
                dqn_solver.remember(state, action, reward, state_next, terminal)
                state = state_next
                if terminal:
                    print("Thread_id: ", self.thread_id, "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    score_logger.add_score(step, run)
                    break
                dqn_solver.experience_replay()


if __name__ == "__main__":
    dqn = DQN()
    dqn.train()
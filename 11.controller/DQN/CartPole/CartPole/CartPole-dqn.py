import math
import gym
import threading
from collections import deque
from logger import get_logger
import numpy as np
import tensorflow as tf
import pylab
import time
from keras.layers import Input, Conv2D, Flatten, Dense, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
import os
import pickle
from CartPole_dqn_worker import Worker
# from A3C_worker_full_steps import WorkerForFullSteps

print(tf.__version__)
THREADS_NUM = 2
series_size = 8 # MLP에서는 사용하지 않음
feature_size = 4
# x : -0.061586
# θ : -0.75893141
# dx/dt : 0.05793238
# dθ/dt : 1.15547541

action_size = 2

model_type = "MLP"
#model_type = "LSTM"
# model_type = "CNN"

load_model = False  # 훈련할 때
#load_model = True    # 훈련이 끝나고 Play 할 때

MAX_EPISODES = 1000

SUCCESS_CONSECUTIVE_THRESHOLD = 5
SAVED_MODEL_PATH = "./save/DQN_rotary_pendulum.h5"
global_logger = get_logger("./cartpole_a3c")
INFO_PATH = "./save_info/"
MODEL_PATH = "./save_model/"
GRAPTH_PATH = "./save_graph/"
LOG_PATH = "./logs/"
MEMORY_SIZE = 2000

class DQN:
    def __init__(self):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 10 ** (-8)
        self.lr = 0.001
        self.discount_factor = .99
        self.hidden1, self.hidden2, self.hidden3 = 100, 100, 16

        self.global_score_list = []
        self.global_actor_loss_list = []
        self.global_critic_loss_list = []
        self.__global_score_list_lock = threading.RLock()

        self.memory = deque(maxlen=MEMORY_SIZE)

        # generate model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if not os.path.exists(INFO_PATH):
            os.makedirs(INFO_PATH)

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        if not os.path.exists(GRAPTH_PATH):
            os.makedirs(GRAPTH_PATH)

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists("~/git/auto_trading/web_app/static/img"):
            os.makedirs("~/git/auto_trading/web_app/static/img")

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.num_consecutive_success = 0
        self.global_episode = 0

    def build_model(self):
        if model_type == "MLP":
            input = Input(batch_shape=(None, feature_size), name="state")
            shared_1 = Dense(units=self.hidden1, activation='relu')(input)
            shared_2 = Dense(units=self.hidden2, activation="relu")(shared_1)
        elif model_type == "LSTM":
            input = Input(batch_shape=(None, series_size, feature_size), name="state")
            shared_1 = LSTM(
                units=self.hidden1,
                input_shape=(series_size, feature_size),  # (타임스텝, 속성)
                activation='relu',
                dropout=0.2,
                return_sequences=True
            )(input)

            shared_2 = LSTM(
                units=self.hidden2,
                activation="relu",
                dropout=0.3,
                return_sequences=False
            )(shared_1)
        else:
            input = Input(batch_shape=(None, series_size, feature_size, 1), name="state")
            shared_1 = Conv2D(
                filters=32,
                kernel_size=(int(math.floor(series_size / 4)), 2),
                activation='relu',
                input_shape=(1, series_size, feature_size)
            )(input)

            shared_2 = Conv2D(
                filters=32,
                kernel_size=(int(math.floor(series_size / 4)), 2),
                activation='relu',
            )(shared_1)

            shared_2 = Flatten()(shared_2)

        hidden = Dense(self.hidden3, activation='relu', kernel_initializer='glorot_normal')(shared_2)
        logit = Dense(action_size, activation='linear', kernel_initializer='glorot_normal')(hidden)

        model = Model(inputs=input, outputs=logit)

        model._make_predict_function()

        model.summary()

        return model

    # update target model into weight of model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        workers = [
            Worker(idx, self, self.model, self.target_model, self.discount_factor,
                  self.sess, series_size, feature_size, action_size, MAX_EPISODES, model_type) for idx in range(THREADS_NUM)
        ]
        for agent in workers:
            agent.start()
        while True:
            if self.num_consecutive_success >= SUCCESS_CONSECUTIVE_THRESHOLD:
                for agent in workers:
                    agent.running = False
                print("SUCCESS!!!")
                break

            is_anyone_alive = True
            for agent in workers:
                is_anyone_alive = agent.is_alive()

            if not is_anyone_alive:
                break

            time.sleep(1)

    def play(self):
        # env = gym.make("CartPole-v0").unwrapped
        env = gym.make("CartPole-v0")
        state = env.reset()

        self.model.load_weights(SAVED_MODEL_PATH)

        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                action = self.get_action(state)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

    # select action based on epsilon greedy policy
    def get_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        #     q_value = self.model.predict(state)
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    @staticmethod
    def exp_moving_average(values, window):
        """ Numpy implementation of EMA
        """
        if window >= len(values):
            sma = np.mean(np.asarray(values))
            a = [sma] * len(values)
        else:
            weights = np.exp(np.linspace(-1., 0., window))
            weights /= weights.sum()
            a = np.convolve(values, weights, mode='full')[:len(values)]
            a[:window] = a[window]
        return a

    def save_model(self):
        with self.__global_score_list_lock:
            self.model.save_weights(SAVED_MODEL_PATH)

    def append_global_score_list(self, idx, episode, score, actor_loss, critic_loss):
        self.global_episode += 1

        with self.__global_score_list_lock:

            if score >= 195 and self.global_score_list[-1] < 195:
                self.num_consecutive_success = 1
            elif score >= 195 and self.global_score_list[-1] >= 195:
                self.num_consecutive_success += 1
            else:
                self.num_consecutive_success = 0

            self.global_score_list.append(score)

            global_logger.info("{0}: {1:>2}-Episode {2:>3d}: SCORE {3:.6f}, ACTOR LOSS: {4:.6f}, CRITIC_LOSS: {5:.6f}, num_consecutive_success: {6}".format(
                self.global_episode,
                idx,
                episode,
                score,
                actor_loss,
                critic_loss,
                self.num_consecutive_success
            ))

            pylab.clf()
            pylab.plot(range(len(self.global_score_list)), self.global_score_list, 'b')
            pylab.plot(range(len(self.global_score_list)), self.exp_moving_average(self.global_score_list, 10), 'r')
            pylab.legend(["Score", "Averaged Score"])
            pylab.xlabel("Episodes")
            pylab.ylabel("Scores")
            pylab.savefig("./save_graph/global_score.png")

            with open('./save_graph/global_score.pickle', 'wb') as f:
                pickle.dump([self.global_score_list, self.exp_moving_average(self.global_score_list, 10)], f)

            self.global_actor_loss_list.append(actor_loss)
            self.global_critic_loss_list.append(critic_loss)

            pylab.clf()
            pylab.plot(range(len(self.global_actor_loss_list)), self.global_actor_loss_list, 'b')
            pylab.plot(range(len(self.global_critic_loss_list)), self.global_critic_loss_list, 'r')
            pylab.yscale('log')
            pylab.legend(["Actor Loss", "Critic Loss"])
            pylab.xlabel("Episodes")
            pylab.ylabel("Losses")

            # global_error = np.asarray(self.global_actor_loss_list) + np.asarray(self.global_critic_loss_list)
            # pylab.plot(range(len(self.global_critic_loss_list)), global_error.tolist(), 'g')
            # pylab.legend(["Actor Loss", "Critic Loss", "Sum Loss"])

            pylab.savefig("./save_graph/global_error.png")

            with open('./save_graph/global_error.pickle', 'wb') as f:
                pickle.dump([self.global_actor_loss_list, self.global_critic_loss_list], f)


if __name__ == "__main__":
    global_dqn = DQN()

    if load_model:
        global_dqn.play()
    else:
        global_dqn.train()

import math
import threading
import gym
from logger import get_logger
import numpy as np
import tensorflow as tf
import pylab
import time
from keras.layers import Input, Conv2D, Flatten, Dense, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
import os
import pickle
from DQN_worker import Worker
import random
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

print(tf.__version__)

series_size = 50 # MLP에서는 사용하지 않음
feature_size = 4
# x : -0.061586
# θ : -0.75893141
# dx/dt : 0.05793238
# dθ/dt : 1.15547541

action_size = 3
DQN_MODEL_PATH = "./save_model/cartpole_dqn_model.h5"
DQN_MEMORY_PATH = "./save_model/cartpole_dqn_memory.h5"
LOGGER_PATH = "./CartPole"
INFO_PATH = "./save_info/"
MODEL_PATH = "./save_model/"
GRAPTH_PATH = "./save_graph/"
LOG_PATH = "./logs/"
GLOBAL_SCORE_PNG = GRAPTH_PATH + 'global_score.png'
GLOBAL_SCORE_PICKLE = GRAPTH_PATH + 'global_score.pickle'
GLOBAL_ERROR_PNG = GRAPTH_PATH + 'global_error.png'
GLOBAL_ERROR_PICKLE = GRAPTH_PATH + 'global_error.pickle'

model_type = "MLP"
# model_type = "LSTM"
# model_type = "CNN"

load_model = False  # 훈련할 때
#load_model = True    # 훈련이 끝나고 Play 할 때

MAX_EPISODES = 5000

SUCCESS_CONSECUTIVE_THRESHOLD = 5

THREADS = 2

global_logger = get_logger('./')

class A3C:
    def __init__(self):
        self.load_model = False

        self.actor_lr = 0.0002 # 0.001
        self.critic_lr = 0.0005# 0.001
        self.rho = 0.95
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 0.001

        self.discount_factor = .99
        self.hidden1, self.hidden2, self.hidden3 = 256, 256, 16 # 100, 100, 16

        self.global_score_list = []
        self.global_actor_loss_list = []
        self.global_critic_loss_list = []
        self.__global_score_list_lock = threading.RLock()

        # create model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

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

        if os.path.exists(DQN_MODEL_PATH):
            self.load_model = True

        if self.load_model:
            self.model.load_weights(DQN_MODEL_PATH)
            self.memory_load(DQN_MEMORY_PATH)

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.num_consecutive_success = 0
        self.global_episode = 0

    def memory_dump(self, file_name):
        with open(file_name, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)

    def memory_load(self, file_name):
        with open(file_name, 'rb') as memory_file:
            self.memory = pickle.load(memory_file)

    def build_model(self):
        if model_type == "MLP":
            input = Input(batch_shape=(None, feature_size * series_size), name="state")
            shared_1 = Dense(units=self.hidden1, activation='elu')(input)
            shared_2 = Dense(units=self.hidden2, activation="elu")(shared_1)
            shared_3 = Dense(units=self.hidden2, activation="elu")(shared_2)
        elif model_type == "LSTM":
            input = Input(batch_shape=(None, series_size, feature_size), name="state")
            shared_1 = LSTM(
                units=self.hidden1,
                input_shape=(series_size, feature_size),  # (타임스텝, 속성)
                activation='elu',
                dropout=0.2,
                return_sequences=True
            )(input)

            shared_2 = LSTM(
                units=self.hidden2,
                activation="elu",
                dropout=0.3,
                return_sequences=False
            )(shared_1)
        else:
            input = Input(batch_shape=(None, feature_size, series_size, 1), name="state")
            model = Sequential([
                Conv2D(32, (4, 4), activation='relu'),
                Conv2D(64, (1, 4), activation='relu'),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(action_size, activation='linear', kernel_initializer='glorot_uniform'),
                Dense(1, activation='linear', kernel_initializer='he_uniform')
            ])

            conv_layer_1 = model.layers[0](input)
            conv_layer_2 = model.layers[1](conv_layer_1)

            flatten = model.layers[2](conv_layer_2)
            dense = model.layers[3](flatten)
            action = model.layers[4](dense)

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

        model._make_predict_function()

        model.summary()

        return model

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, action_size), name="action")
        advantages = K.placeholder(shape=(None,), name="advantages")
        logits = self.actor.output
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(action, axis=1),
                                                                     logits=logits)

        policy_loss *= tf.stop_gradient(advantages)
        actor_loss = policy_loss - 0.01 * entropy

        # action = K.placeholder(shape=(None, action_size), name="action")
        # advantages = K.placeholder(shape=(None, 1), name="advantages", dtype=tf.float32)
        # logits = self.actor.output
        # policy = tf.nn.softmax(logits)
        # xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
        # policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(action, axis=1),
        #                                                              logits=logits)
        # policy_loss *= tf.stop_gradient(advantages)
        # actor_loss = policy_loss - 0.01 * xentropy

        # optimizer = Adam(lr=self.actor_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
        # optimizer = RMSprop(lr=self.actor_lr, rho=self.rho, epsilon=self.epsilon)
        optimizer = SGD(lr=self.actor_lr, decay=1e-6, momentum=0.9, nesterov=True)


        with self.__global_score_list_lock:
            updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)

        weights = self.actor.trainable_weights[-1]

        train = K.function(
            [self.actor.input, action, advantages],
            [actor_loss, policy, weights],
            updates=updates
        )

        global_logger.info("action: {0}".format(action))
        global_logger.info("advantages: {0}".format(advantages))
        global_logger.info("policy: {0}".format(policy))
        global_logger.info("entropy: {0}".format(entropy))
        global_logger.info("policy_loss: {0}".format(policy_loss))
        global_logger.info("actor_loss: {0}".format(actor_loss))
        global_logger.info("self.actor.trainable_weights: {0}".format(self.actor.trainable_weights))

        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1), name="discounted_reward")
        value = self.critic.output
        critic_loss = tf.reduce_mean(tf.square(discounted_reward - value))

        # optimizer = Adam(lr=self.critic_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
        # optimizer = RMSprop(lr=self.actor_lr, rho=self.rho, epsilon=self.epsilon)
        optimizer = SGD(lr=self.actor_lr, decay=1e-6, momentum=0.9, nesterov=True)


        # update
        with self.__global_score_list_lock:
            updates = optimizer.get_updates(self.critic.trainable_weights, [], critic_loss)

        weights_0 = self.actor.trainable_weights[0]
        weights_1 = self.actor.trainable_weights[1]
        weights_2 = self.actor.trainable_weights[2]
        weights_3 = self.actor.trainable_weights[3]

        train = K.function(
            [self.critic.input, discounted_reward],
            [critic_loss, value, weights_0, weights_1, weights_2, weights_3],
            updates=updates
        )

        global_logger.info("discounted_reward: {0}".format(discounted_reward))
        global_logger.info("critic ourput value: {0}".format(value))
        global_logger.info("self.critic.trainable_weights: {0}".format(self.critic.trainable_weights))

        return train

    def train(self):
        workers = [
            Worker(idx, self, self.actor, self.critic, self.optimizer, self.discount_factor,
                   self.sess, series_size, feature_size, action_size, MAX_EPISODES, model_type) for idx in
            range(THREADS)
        ]

        for agent in workers:
            time.sleep(1)
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

    # def play(self):
    #     global ENV_COUNT
    #     env = None
    #
    #     if ENV_COUNT == 0:
    #         # env = self.env_1
    #         env = ENV_1()
    #         ENV_COUNT += 1
    #     elif ENV_COUNT == 1:
    #         # env = self.env_2
    #         env = ENV_2()
    #
    #     state = env.reset()
    #
    #     state = np.reshape(state, [1, feature_size])
    #
    #     # make init history
    #     history = np.array([state for _ in range(HISTORY_LENGTH)])
    #
    #     if os.path.exists(ACTOR_MODEL_PATH) and os.path.exists(CRITIC_MODEL_PATH):
    #         self.load_model()
    #
    #     done = False
    #     step_counter = 0
    #     reward_sum = 0
    #
    #     try:
    #         while not done:
    #             env.render(mode='rgb_array')
    #             action = self.get_action([history])
    #             state, reward, done, _ = env.step(action)
    #             reward_sum += reward
    #             print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
    #             step_counter += 1
    #     except KeyboardInterrupt:
    #         print("Received Keyboard Interrupt. Shutting down.")
    #     finally:
    #         env.close()

    def get_action(self, state):
        if model_type == "MLP":
            policy = self.actor.predict(np.reshape(state, [1, series_size*feature_size]))[0]
        elif model_type == "LSTM":
            policy = self.actor.predict(np.reshape(state, [1, series_size, feature_size]))[0]
        else:
            policy = self.actor.predict(np.reshape(state, [1, series_size, feature_size, 1]))[0]
        return np.random.choice(action_size, 1, p=policy)[0]

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
            self.actor.save_weights(ACTOR_MODEL_PATH)
            self.critic.save_weights(CRITIC_MODEL_PATH)

    def load_model(self):
        self.actor.load_weights(DQN_MODEL_PATH)

    def append_global_score_list(self, idx, episode, score, actor_loss, critic_loss):
        self.global_episode += 1

        with self.__global_score_list_lock:

            if score >= 5000 and self.global_score_list[-1] < 5000:
                self.num_consecutive_success = 1
            elif score >= 5000 and self.global_score_list[-1] >= 5000:
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
            pylab.plot(range(len(self.global_score_list)), self.global_score_list, 'r', alpha=0.3)
            pylab.plot(range(len(self.global_score_list)), self.exp_moving_average(self.global_score_list, 10), 'r')
            pylab.legend(["Score", "Averaged Score"])
            pylab.xlabel("Episodes")
            pylab.ylabel("Scores")
            pylab.savefig(GLOBAL_SCORE_PNG)

            with open(GLOBAL_SCORE_PICKLE, 'wb') as f:
                pickle.dump([self.global_score_list, self.exp_moving_average(self.global_score_list, 10)], f)

            self.global_actor_loss_list.append(actor_loss)
            self.global_critic_loss_list.append(critic_loss)

            pylab.clf()
            pylab.plot(range(len(self.global_actor_loss_list)), self.global_actor_loss_list, 'royalblue', alpha=0.3)
            pylab.plot(range(len(self.global_critic_loss_list)), self.global_critic_loss_list, 'darkorange', alpha=0.3)
            pylab.plot(range(len(self.global_actor_loss_list)), self.exp_moving_average(self.global_actor_loss_list, 10), 'royalblue')
            pylab.plot(range(len(self.global_critic_loss_list)), self.exp_moving_average(self.global_critic_loss_list, 10), 'darkorange')

            pylab.yscale('log')
            pylab.legend(["Actor Loss", "Critic Loss", "Averaged Actor Loss", "Averaged Critic Loss"])
            pylab.xlabel("Episodes")
            pylab.ylabel("Losses")

            # global_error = np.asarray(self.global_actor_loss_list) + np.asarray(self.global_critic_loss_list)
            # pylab.plot(range(len(self.global_critic_loss_list)), global_error.tolist(), 'g')
            # pylab.legend(["Actor Loss", "Critic Loss", "Sum Loss"])

            pylab.savefig(GLOBAL_ERROR_PNG)

            with open(GLOBAL_ERROR_PICKLE, 'wb') as f:
                pickle.dump([self.global_actor_loss_list, self.global_critic_loss_list], f)


if __name__ == "__main__":
    global_a3c = A3C()

    if load_model:
        global_a3c.play()
    else:
        global_a3c.train()

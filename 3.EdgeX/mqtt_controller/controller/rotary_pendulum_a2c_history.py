from environment3 import Env
import pylab
import math
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from datetime import datetime
import sys
import os.path

import random
import time

K.clear_session()
EPISODES = 20000

class A2CAgent:
    def __init__(self, state_size, action_size, history_size):
        self.load_model = False
        if os.path.exists("./save/pendulum_actor.h5"):
            self.load_model = True

        # 상태와 행동의 크기 정의
        self.state_size = state_size * history_size
        self.action_size = action_size
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weights("./save/pendulum_actor.h5")
            self.critic.load_weights("./save/pendulum_critic.h5")

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done, episode):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        if not self.load_model:
            self.actor_updater([state, act, advantage])
            self.critic_updater([state, target])


if __name__ == "__main__":
    print(str(datetime.now()) + ' started')
    env = Env()

    state_size = env.state_space_shape[0]
    action_size = env.action_space_shape[0]
    history_size = 4

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size, history_size)

    scores, episodes, steps = [], [], []

    episode_start_num = 0
    if os.path.exists("./save/lastest_episode_num.txt"):
        with open("./save/lastest_episode_num.txt", 'r') as f:
            episode_start_num = int(f.readline())

    try:
        for episode in range(episode_start_num, EPISODES):
            done = False
            score = 0
            step = 0

            state = env.reset()

            history = np.stack((state, state, state, state), axis=1)
            history = np.reshape(history, (1, state_size, history_size))

            while not done:
                flat_history = np.reshape(history, (1, state_size * history_size))
                action_index = agent.get_action(flat_history)

                next_state, reward, done, info = env.step(action_index)
                next_state = np.reshape(next_state, [1, state_size, 1])

                next_history = np.append(next_state, history[:, :, :history_size - 1], axis=2)
                flat_next_history = np.reshape(next_history, (1, state_size * history_size))

                step += 1

                rad = math.acos(next_state[0][0])
                print("episode:{0} || time:{1} || step:{2} || action:{3} || pendulum radian:{4} || reward:{5} || done:{6}"
                    .format(
                    episode,
                    datetime.utcnow().strftime('%H-%M-%S.%f')[:-3],
                    step,
                    action_index,
                    # round(next_state[0][0], 4),
                    rad,
                    round(reward, 2),
                    done
                ))

                agent.train_model(flat_history, action_index, reward, flat_next_history, done, episode)

                score += reward
                history = next_history

                # now = datetime.now()
                # print(now - last)
                # last = now
                ###############################################
                ## do not train time = 0.119 (0.118 - 0.121) ##
                ## 1 step train time = 0.126 (0.121 - 0.13)  ##
                ###############################################

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save/pendulum_a2c_history.png")
                    print()
                    print("*** episode:{0} is Done!!! || score:{1} || step:{2} || info:{3} ***".format(
                        episode, score, step, info
                    ))
                    steps.append(step)
                    print()
                    sys.stdout.flush()

                    #env.episode_reset()

                    if episode % 50 == 0:
                        print("average of recent 50 steps :", str(np.mean(steps[-50:])))
                        print()
                        sys.stdout.flush()
                        agent.actor.save_weights("./save/pendulum_actor.h5")
                        agent.critic.save_weights("./save/pendulum_critic.h5")

                        f = open("./save/lastest_episode_num.txt", 'w')
                        f.write(str(episode)+"\n")
                        f.close()

                    # 이전 10개 에피소드의 점수 평균이 -30보다 크면 학습 중단
                    #if np.mean(scores[-min(10, len(scores)):]) > -30:
                    #    agent.actor.save_weights("./save/pendulum_actor.h5")
                    #    agent.critic.save_weights("./save/pendulum_critic.h5")
                    #    env.close()
                    #    sys.exit()
        env.close()
        sys.exit()

    except KeyboardInterrupt as e:
        env.close()
        sys.exit()

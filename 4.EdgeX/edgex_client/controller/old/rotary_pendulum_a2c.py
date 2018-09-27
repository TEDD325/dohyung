from environment import Env
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
import random
import time

K.clear_session()
EPISODES = 2000

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.00001
        self.critic_lr = 0.00001
        self.initial_train_episodes = 5

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()
        self.actor_updater2 = self.actor_optimizer2()
        self.critic_updater2 = self.critic_optimizer2()

        if self.load_model:
            self.actor.load_weights("./save/pendulum_actor.h5")
            self.critic.load_weights("./save/pendulum_critic.h5")

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(48, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(48, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(48, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(48, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(48, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        if np.isnan(policy[0]):
            action = random.randrange(0, 11)
            print(state, "---", policy, "---", action)
            pass
        else:
            action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = RMSprop(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = RMSprop(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer2(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = RMSprop(lr=self.actor_lr * 50)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer2(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = RMSprop(lr=self.critic_lr * 50)
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

        if episode < self.initial_train_episodes and self.load_model == False:
            self.actor_updater([state, act, advantage])
            self.critic_updater([state, target])
        else:
            self.actor_updater2([state, act, advantage])
            self.critic_updater2([state, target])


if __name__ == "__main__":
    print(str(datetime.now()) + ' started')
    env = Env()

    state_size = env.state_space_shape[0]
    action_size = env.action_space_shape[0]

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

    scores, episodes, steps = [], [], []
    try:
        for episode in range(EPISODES):
            done = False
            score = 0
            step = 0

            state = env.reset()
            state = np.reshape(state, [1, state_size])
            while not done:
                # if episode < agent.initial_train_episodes and agent.load_model == False:              # explore in the first episode
                #     if step % 4 < 2:
                #         action_index = random.randrange(6, 11)
                #     else:
                #         action_index = random.randrange(0, 5)
                # else:
                #     action_index = agent.get_action(state)
                action_index = agent.get_action(state)

                next_state, reward, done, info = env.step(action_index)
                next_state = np.reshape(next_state, [1, state_size])
                step += 1

                rad = math.acos(next_state[0][0])
                print("episode:{0} || step:{1} || action:{2} || pendulum radian:{3} || reward:{4} || done:{5}".format(
                    episode,
                    step,
                    action_index,
                    # round(next_state[0][0], 4),
                    rad,
                    round(reward, 2),
                    done
                ))

                agent.train_model(state, action_index, reward, next_state, done, episode)

                score += reward
                state = next_state

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
                    pylab.savefig("./save/pendulum_a2c.png")
                    print()
                    print("*** episode:{0} is Done!!! || score:{1} || step:{2} || info:{3} ***".format(
                        episode, score, step, info
                    ))
                    steps.append(step)
                    print()

                    #env.episode_reset()

                    if episode % 50 == 0:
                        print("average of recent 50 steps :", str(np.mean(steps[-50:])))
                        print()
                    #if -10 <= score:
                        #agent.actor.save_weights("./save/pendulum_actor.h5")
                        #agent.critic.save_weights("./save/pendulum_critic.h5")
                        #max_score = score
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

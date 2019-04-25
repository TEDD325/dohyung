import tensorflow as tf
import gym
import numpy as np
import random as ran


env = gym.make('CartPole-v1')

N_REPLAY_ELEMENT = 10 # 꺼낼 리플레이 요소 개수
N_MEMORY_ELEMENT = 50000
N_MINIBATCH = 50
N_EPISODE = 2000
N_NEURON = 10
N_STEP_FOR_TRAINING = 10
N_STATE = env.observation_space.shape[0] # 4
N_ACTION = env.action_space.n # 2
REPLAY_MEMORY = []
LEARNING_RATE = 0.01
DISCOUNT = 0.99


x = tf.placeholder(dtype=tf.float32,
                   shape=(1,N_STATE))
W1 = tf.get_variable(name='W1',
                     shape=[N_STATE,N_NEURON],
                     initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name='W2',
                     shape=[N_NEURON, N_ACTION],
                     initializer=tf.contrib.layers.xavier_initializer())
output1 = tf.matmul(x, W1)
L1 = tf.nn.tanh(output1)
Q_pred = tf.matmul(L1, W2)
'''
L1: [1, N_NEURON]
W2: [N_NEURON, N_ACTION]
Q_pred: [1, N_ACTION]
'''
y = tf.placeholder(dtype=tf.float32,
                   shape=(1, N_ACTION))

squared_error = tf.square(y-Q_pred)
loss = tf.reduce_sum(squared_error)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

reward_List = []

with tf.Session() as sess:
    sess.run(init)

    for episode in range(N_EPISODE):
        initial_state = env.reset()
        # print("initial_state:", initial_state)
        # print("type(initial_state):", type(initial_state))
        # print("initial_state.shape:", initial_state.shape)
        epsilon = 1.0 / ((episode/25)+1)
        reward_all = 0
        done = False
        count = 0

        while not done:
            count += 1

            current_state = np.reshape(initial_state,
                                       newshape=[1, N_STATE])
            Q_value = sess.run(Q_pred,
                               feed_dict={x: current_state})

            if epsilon > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_value) # Q_value: [1, N_ACTION]

            next_state, reward, done, info = env.step(action)

            REPLAY_MEMORY.append([current_state,
                                  action,
                                  reward,
                                  next_state,
                                  done])

            if len(REPLAY_MEMORY) > N_MEMORY_ELEMENT:
                del REPLAY_MEMORY[0]

            reward_all += reward
            initial_state = next_state

        if episode % N_STEP_FOR_TRAINING == 1:
            for i in range(N_MINIBATCH):
                for sample in ran.sample(REPLAY_MEMORY, N_REPLAY_ELEMENT):
                    current_state_, action_, reward_, next_state_, done_ = sample

                    if done_:
                        Q_value[0, action_] = -100
                    else:
                        next_state_pred = np.reshape(next_state_,
                                                     newshape=[1, N_STATE])
                        Q_value_pred = sess.run(Q_pred,
                                                feed_dict = {x: next_state_pred})
                        Q_value[0, action_] = reward_ + DISCOUNT*np.max(Q_value_pred)
                        loss = Q_value_pred - Q_pred
                        print("loss:", loss)
                    sess.run(train,
                             feed_dict = {x: current_state_, y: Q_value})
        reward_List.append(reward_all)
        print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(episode,
                                                                                           count,
                                                                                           reward_all,
                                                                                           np.mean(reward_List))
              )


    # for episode in range(500):
    #     # state 초기화
    #     initial_state = env.reset()
    #
    #     reward_all = 0
    #     done = False
    #     count = 0
    #     # 에피소드가 끝나기 전까지 반복
    #     while not done:
    #         env.render()
    #         count += 1
    #         # state 값의 전처리
    #         current_state = np.reshape(initial_state,
    #                                    newshape=[1, N_STATE])
    #
    #         # 현재 상태의 Q값을 에측
    #         Q_value = sess.run(Q_pred,
    #                            feed_dict={x: current_state})
    #         action = np.argmax(Q_value)
    #
    #         # 결정된 action으로 Environment에 입력
    #         initial_state, reward, done, info = env.step(action)
    #
    #         # 총 reward 합
    #         reward_all += reward
    #
    #     reward_List.append(reward_all)
    #
    #     print("Episode : {} steps : {} r={}. averge reward : {}".format(episode,
    #                                                                     count,
    #                                                                     reward_all,
    #                                                                     np.mean(reward_List))
    #           )

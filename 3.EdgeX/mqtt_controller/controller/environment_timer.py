import threading
import math
from collections import deque
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np
import json

MQTT_SERVER = 'localhost'

MQTT_MOTOR_POWER = 'motor_power'
MQTT_MOTOR_RESET = 'motor_reset_position'

MQTT_PENDULUM_STATE = 'pendulum_state_info'
MQTT_MOTOR_STATE = 'motor_state_info'
MQTT_MOTOR_RESET_COMPLETE = 'reset_complete'

sub_topic_list = [MQTT_PENDULUM_STATE, MQTT_MOTOR_STATE]

motor_speed_list = [-100, -70, -50, -30, -10, 0, 10, 30, 50, 70, 100]
# motor_speed_list = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]

self_env = None


class Env:
    def __init__(self):
        global self_env

        self_env = self

        self.state_space_shape = (30,)
        self.action_space_shape = (11,)
        self.state_list = []

        self.list_size = 5
        self.pendulum_info_list = [[]] * self.list_size
        self.motor_info_list = [[]] * self.list_size
        self.pendulum_state = []
        self.motor_state = []

        self.last_pendulum_radian = 0
        self.last_motor_radian = 0
        self.last_pendulum_time = 0
        self.last_motor_time = 0

        self.sub = mqtt.Client(client_id="env_sub", transport="TCP")
        self.sub.on_connect = self.__on_connect
        self.sub.on_message = self.__on_message
        self.sub.username_pw_set(username="link", password="0123")
        self.sub.connect(MQTT_SERVER, 1883, 60)

        sub_thread = threading.Thread(target=self.__sub, args=(self.sub,))
        sub_thread.daemon = True
        sub_thread.start()

        self.pub = mqtt.Client(client_id="env_pub", transport="TCP")
        self.pub.username_pw_set(username="link", password="0123")
        self.pub.connect(MQTT_SERVER, 1883, 60)

        self.steps = 0
        self.done = False
        self.isResetComplete = False
        self.reward = 0
        self.rewards = []
        self.info = None

    @staticmethod
    def __on_connect(client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc))
        client.subscribe(topic=MQTT_PENDULUM_STATE)
        client.subscribe(topic=MQTT_MOTOR_STATE)
        client.subscribe(topic=MQTT_MOTOR_RESET_COMPLETE)

    @staticmethod
    def __sub(sub):
        try:
            print("Sub thread started!")
            sub.loop_forever()
        except KeyboardInterrupt as e:
            print("Sub thread Interrupted: {0}.format(0)")
            sub.unsubscribe(sub_topic_list)

    @staticmethod
    def __on_message(client, userdata, msg):
        #print(msg.topic + ': {0}'.format(datetime.utcnow().strftime('%H-%M-%S.%f')[:-3]))
        if msg.topic == MQTT_PENDULUM_STATE:
            m_decode = str(msg.payload.decode("utf-8"))
            pendulum_info = json.loads(m_decode)['angle']
            self_env.__insert_angle_info_to_buffer('pendulum', pendulum_info) #pendulum_info is only angle value

        if msg.topic == MQTT_MOTOR_STATE:
            motor_info = str(msg.payload.decode("utf-8")).split('|')
            self_env.__insert_angle_info_to_buffer('motor', motor_info)

        if msg.topic == MQTT_MOTOR_RESET_COMPLETE:
            print("***** Reset Complete! *****")
            self_env.isResetComplete = True

    def __insert_angle_info_to_buffer(self, device, info):
        if device == 'pendulum':
            pendulum_radian = math.radians(int(info))

            pendulum_cosine_theta = math.cos(float(pendulum_radian))
            pendulum_sine_theta = math.sin(float(pendulum_radian))

            angular_variation = (pendulum_radian - self.last_pendulum_radian)
            # angular variation filtering
            if angular_variation > 2.5:
                angular_variation -= math.pi * 2
            elif angular_variation < -2.5:
                angular_variation += math.pi * 2

            pendulum_angular_velocity = angular_variation / 0.015
            self.last_pendulum_radian = pendulum_radian

            # pendulum angle, pendulum angular velocity
            # print(info[0], pendulum_angular_velocity/5.)

            self.pendulum_info_list.append([pendulum_radian,
                                            [pendulum_cosine_theta, pendulum_sine_theta, pendulum_angular_velocity/5.]
                                            ])
            self.pendulum_info_list = self.pendulum_info_list[-5:]
        elif device == 'motor':
            motor_radian = math.radians(int(info[0]))
            motor_state_time = info[1]

            motor_cosine_theta = math.cos(float(motor_radian))
            motor_sine_theta = math.sin(float(motor_radian))

            motor_angular_velocity = (motor_radian - self.last_motor_radian) / 0.015
            self.last_motor_radian = motor_radian

            self.motor_info_list.append([motor_radian,
                                        motor_state_time,
                                        [motor_cosine_theta, motor_sine_theta, motor_angular_velocity/10.]
                                         ])
            self.motor_info_list = self.motor_info_list[-5:]
        else:
            raise AttributeError("Unrecognized Device")

    def reset(self):
        # reset position
        self.steps = 0
        self.reward = 0
        self.done = False
        self.isResetComplete = False
        del self.rewards[:]
        del self.state_list[:]

        # reset position
        self.pub.publish(topic=MQTT_MOTOR_RESET, payload=MQTT_MOTOR_RESET)
        while not self.isResetComplete:
            time.sleep(0.05)

        # pendulum_info_list = [radian, time, [cos(radian), sin(radian), angular_velocity]]
        self.pendulum_state = self.pendulum_info_list
        # motor_info_list = [radian, time, [cos(radian), sin(radian)]]
        self.motor_state = self.motor_info_list

        # reshape state (total 30 values)
        for i in range(self.list_size):
            print(self.pendulum_state)
            self.state_list += self.pendulum_state[i][1]
            self.state_list += self.motor_state[i][2]

        return self.state_list

    def step(self, action_index):
        self.steps += 1

        del self.state_list[:]

        # motor_speed = (action_index - 5) * 20
        motor_speed = motor_speed_list[action_index]

        # set action
        self.pub.publish(topic=MQTT_MOTOR_POWER, payload=str(motor_speed))

        # pendulum_info_list = [radian, time, [cos(radian), sin(radian), angular_velocity]]
        self.pendulum_state = self.pendulum_info_list
        # motor_info_list = [radian, time, [cos(radian), sin(radian)]]
        self.motor_state = self.motor_info_list

        ################################################################################################################
        ####### OpenAI Gym reward = -(pendulum_radian^2 + 0.1 * pendulum angular velocity^2 + 0.0001 * action^2) #######
        ################################################################################################################
        # reward = -(pendulum_radian ^ 2 + 0.1 * pendulum angular velocity ^ 2 + 0.0001 * action ^ 2)
        self.reward = -((self.pendulum_state[4][0] ** 2)
                        + (0.1 * (self.pendulum_state[4][1][2] ** 2))
                        + (0.01 * abs(motor_speed) * (math.pi - abs(self.pendulum_state[4][0]))))
        # self.reward = -((self.pendulum_state[4][0] ** 2)
        #                 + (abs(self.pendulum_state[4][2][2]))
        #                 + (0.01 * ((motor_speed/10.0) ** 2)))
        #print(self.pendulum_state[4][0] ** 2)
        #print(abs(self.pendulum_state[4][2][2]))
        #print(0.01 * ((motor_speed/10.0) ** 2))
        #print("------------------------------------")
        #print(self.reward)

        ################################################################################################################

        #self.reward = -(abs(self.pendulum_state[4][0] * 10)
        #                + (abs(self.pendulum_state[4][2][2])) / (abs(self.pendulum_state[4][0]) + 0.1)
        #                + abs(motor_speed) / (abs(self.pendulum_state[4][0]) + 1))

        #print("|| {0:5d} || {1:5.3f} || {2:5d} ||     || {3:5.3f} || {4:5.3f} || {5:5.3f} ||     || {6:6.3f} ||".format(
        #    self.pendulum_state[4][1],
        #    abs(self.pendulum_state[4][2][2]),
        #    motor_speed,
        #    abs(self.pendulum_state[4][0] * 10),
        #    (abs(self.pendulum_state[4][2][2])) / (abs(self.pendulum_state[4][0]) + 0.1),
        #    abs(motor_speed) / (abs(self.pendulum_state[4][0]) + 1),
        #    self.reward
        #))

        ################################################################################################################

        self.rewards.append(self.reward)

        self.isDone()

        if self.done:
            self.pub.publish(topic=MQTT_MOTOR_POWER, payload=str(0))

        # reshape state (total 30 values)
        for i in range(self.list_size):
            self.state_list += self.pendulum_state[i][1]
            self.state_list += self.motor_state[i][2]

        # pendulum angular velocity, motor angular velocity
        # print(self.pendulum_state[4][2][2], self.motor_state[4][2][2])

        # return state, reward, done, info
        return self.state_list, self.reward, self.done, self.info

    def close(self):
        self.pub.publish(topic=MQTT_MOTOR_RESET, payload=MQTT_MOTOR_RESET)   # set motor speed 0

    def isDone(self):
        if self.steps >= 200:                      # maximum step
            self.info = "max step"
            self.done = True
        elif np.mean(self.rewards[-100:]) > -1:    # success
            self.info = "success"
            self.done = True
        else:
            self.done = False


# if __name__ == "__main__":
#     try:
#         env = Env()
#         env.reset()
#         while True:
#             env.step(2)
#             time.sleep(0.5)
#             env.step(8)
#             time.sleep(0.5)
#
#     except KeyboardInterrupt as e:
#         env.close()
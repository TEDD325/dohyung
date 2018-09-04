import threading
import math
from enum import Enum
from collections import deque
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np
import sys

MQTT_SERVER = 'localhost'

MQTT_MOTOR_RESET = 'reset'
MQTT_MOTOR_RECTIFY = 'rectify'
MQTT_MOTOR_POWER = 'motor_power'

MQTT_MOTOR_RESET_COMPLETE = 'reset_complete'
MQTT_MOTOR_AND_PENDULUM_STATE = 'state_info'

sub_topic_list = [MQTT_MOTOR_RESET_COMPLETE, MQTT_MOTOR_AND_PENDULUM_STATE]
pub_topic_list = [MQTT_MOTOR_RESET, MQTT_MOTOR_RECTIFY, MQTT_MOTOR_POWER]

self_env = None
isResetComplete = False
isRectifyComplete = False
state_changed = False


class Env:
    def __init__(self):
        global self_env
        global isResetComplete
        global isRectifyComplete
        global state_changed

        self_env = self

        self.pendulum_info_buffer = deque()
        self.motor_info_buffer = deque()
        self.state_space_shape = (5,)
        self.action_space_shape = (11,)

        self.sub = mqtt.Client(client_id="env_sub", transport="TCP")
        self.sub.on_connect = self.__on_connect
        self.sub.on_message = self.__on_message
        self.sub.connect(MQTT_SERVER, 1883, 60)

        s = threading.Thread(target=self.__sub, args=(self.sub,))
        s.daemon = True
        s.start()

        self.pub = mqtt.Client(client_id="env_pub", transport="TCP")
        self.pub.connect(MQTT_SERVER, 1883, 60)

        self.steps = 0
        self.done = False
        self.reward = 0
        self.rewards = []
        self.info = None

        self.state_info = dict()
        self.state_info['state'] = list()
        self.state_info['motor_rad'] = None
        self.state_info['pendulum_rad'] = None

    @staticmethod
    def __on_connect(client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc))
        client.subscribe(topic=MQTT_MOTOR_RESET_COMPLETE)
        client.subscribe(topic=MQTT_MOTOR_AND_PENDULUM_STATE)

    @staticmethod
    def __sub(sub):
        try:
            print("Sub thread started!")
            sub.loop_forever()
        except KeyboardInterrupt:
            print("Sub Interrupted!")
            sub.unsubscribe(sub_topic_list)

    @staticmethod
    def __on_message(client, userdata, msg):
        global isResetComplete
        global isRectifyComplete
        global state_changed
        print(msg.topic)
        if msg.topic == MQTT_MOTOR_RESET_COMPLETE:
            isResetComplete = True

        if msg.topic == MQTT_MOTOR_AND_PENDULUM_STATE:
            state_info = str(msg.payload.decode("utf-8")).split('|')
            self_env.state_info['state'], self_env.state_info['motor_rad'], self_env.state_info['pendulum_rad'] \
                = self_env.calculate_state(state_info)
            isRectifyComplete = True
            state_changed = True

        #     msg = pendulum_info.split(":")
        #     theta = int(msg[1].split(',')[0])
        #     rpm = int(msg[2].split("'")[0])
        #     self_env.__insert_angle_info_to_buffer(device=Device.pendulum, theta=theta, rpm=rpm)
        #     #print("Pendulum Angle value: {0}, {1}".format(angle, speed))
        # elif msg.topic == MQTT_MOTOR_ANGLE_TOPIC:
        #     motor_info = str(msg.payload)
        #     msg = motor_info.split(":")
        #     theta = int(msg[1].split("'")[0])
        #     self_env.__insert_angle_info_to_buffer(device=Device.motor, theta=theta, rpm=0)
        #     #print("Motor Angle value: {0}, {1}".format(angle, speed))
        # elif True: # Reset Complete Test
        #     pass

    def calculate_state(self, state_info):
        pendulum_rad = math.radians(int(state_info[0]))
        pendulum_rpm = int(state_info[1])
        motor_rad = math.radians(int(state_info[2]))
        pendulum_cosine_theta = math.cos(float(pendulum_rad))
        pendulum_sine_theta = math.sin(float(pendulum_rad))
        motor_cosine_theta = math.cos(float(motor_rad))
        motor_sine_theta = math.sin(float(motor_rad))
        state = [pendulum_cosine_theta, pendulum_sine_theta, pendulum_rpm, motor_cosine_theta, motor_sine_theta]
        return (state, motor_rad, pendulum_rad)

    def reset(self):
        global isResetComplete
        global isRectifyComplete
        isResetComplete = False
        isRectifyComplete = False

        # reset position
        self.steps = 0
        self.reward = 0
        self.done = False
        del self.rewards[:]

        # reset position
        self.pub.publish(topic=MQTT_MOTOR_RESET, payload=MQTT_MOTOR_RESET)
        while not isResetComplete:
            time.sleep(0.05)

        #rectify
        self.pub.publish(topic=MQTT_MOTOR_RECTIFY, payload=MQTT_MOTOR_RECTIFY)
        while not isRectifyComplete:
            time.sleep(0.001)

        return self.state_info['state']


    def step(self, action_index):
        global state_changed
        state_changed = False
        motor_speed = (action_index - 5) * 20
        self.steps += 1

        # set action
        self.pub.publish(topic=MQTT_MOTOR_POWER, payload=str(motor_speed))
        while not state_changed:
            time.sleep(0.001)

        # action normalization (-2 ~ 2)
        normalized_action = motor_speed / 50.0
        # reward = -(pendulum_radian^2 + 0.1 * pendulum_theta_dot^2 + 0.01 * action^2)
        self.reward = -((self.state_info['pendulum_rad'] ** 2) + (0.1 * (self.state_info['state'][2] / 20 ** 2))
                        + (0.01 * (normalized_action ** 2)))

        self.rewards.append(self.reward)
        self.isDone()

        # return state, reward, done, info
        return self.state_info['state'], self.reward, self.done, self.info

    def close(self):
        self.pub.publish(topic=MQTT_MOTOR_RESET, payload=MQTT_MOTOR_RESET)   # set motor speed 0

    def isDone(self):
        # if pendulum_angle < -0.436 or pendulum_angle > 0.436:
        if self.state_info['pendulum_rad'] < -1.57 or self.state_info['pendulum_rad'] > 1.57:
            self.info = "fall down: " + str(self.state_info['pendulum_rad'])
            self.done = True
        elif self.steps > 300:                        # maximum step
            self.info = "max step"
            self.done = True
        elif np.mean(self.rewards[-30:]) > 20:      # success
            self.info = "success"
            self.done = True
        # elif self.state_info['motor_rad'] > 3.141592 * 5 or self.state_info['motor_rad'] < -3.141592 * 5:  # limit motor position
        #     self.info = "limit position, pos:" + str(round(self.state_info['motor_rad'], 4))
        #     self.reward = -100
        #     self.done = True
        else:
            self.done = False
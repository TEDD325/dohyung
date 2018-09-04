import threading
import math
from collections import deque
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np

MQTT_SERVER = 'localhost'

MQTT_PENDULUM_STATE = 'pendulum_state_info'
MQTT_MOTOR_STATE = 'motor_state_info'

sub_topic_list = [MQTT_PENDULUM_STATE, MQTT_MOTOR_STATE]

self_env = None


class Env:
    def __init__(self):
        global self_env

        self_env = self

        self.state_space_shape = (6,)
        self.action_space_shape = (11,)

        self.last_pendulum_time = 0
        self.last_motor_time = 0

        self.pendulum_info_buffer = deque(maxlen=10)
        self.motor_info_buffer = deque(maxlen=10)
        self.pendulum_info = []
        self.motor_info = []
        self.last_pendulum_angle = 0
        self.last_motor_angle = 0

        self.sub = mqtt.Client(client_id="env_sub", transport="TCP")
        self.sub.on_connect = self.__on_connect
        self.sub.on_message = self.__on_message
        self.sub.connect(MQTT_SERVER, 1883, 60)

        s = threading.Thread(target=self.__sub, args=(self.sub,))
        s.daemon = True
        s.start()

        self.motor_speed = 0
        self.steps = 0
        self.done = False
        self.reward = 0
        self.rewards = []
        self.info = None

    @staticmethod
    def __on_connect(client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc))
        client.subscribe(topic=MQTT_PENDULUM_STATE)
        client.subscribe(topic=MQTT_MOTOR_STATE)

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
        if msg.topic == MQTT_PENDULUM_STATE:
            pendulum_info = str(msg.payload.decode("utf-8")).split('|')
            pendulum_info[1] = float(pendulum_info[1])

            pendulum_step_time = pendulum_info[1] - self_env.last_pendulum_time

            print("|| pendulum_angle: {0} || pendulum_step_time: {1:.4f} ||".format(
                pendulum_info[0], pendulum_step_time
            ))

            self_env.last_pendulum_time = pendulum_info[1]

        if msg.topic == MQTT_MOTOR_STATE:
            motor_info = str(msg.payload.decode("utf-8")).split('|')
            motor_info[1] = float(motor_info[1])

            motor_step_time = motor_info[1] - self_env.last_motor_time

            print("|| motor_angle: {0} || motor_step_time: {1:.4f} || ".format(
                motor_info[0], motor_step_time
            ))

            self_env.last_motor_time = motor_info[1]

if __name__ == "__main__":
    env = Env()

    while True:
        time.sleep(1)

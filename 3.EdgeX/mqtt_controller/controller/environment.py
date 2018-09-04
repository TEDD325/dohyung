import threading
import math
from enum import Enum
from collections import deque
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np

MQTT_SERVER = 'localhost'
MQTT_PENDULUM_ANGLE_TOPIC = 'Pendulum/angle'
MQTT_MOTOR_ANGLE_TOPIC = 'Motor/angle'
MQTT_MOTOR_POWER_TOPIC = 'Motor/power'

class Device(Enum):
    pendulum = 1
    motor = 2

self_env = None

class Env:
    def __init__(self):
        global self_env
        self.pendulum_info_buffer = deque()
        self.motor_info_buffer = deque()
        self.state_space_shape = (5,)
        self.action_space_shape = (11,)

        self.sub = mqtt.Client(client_id="angle_sub", transport="TCP")
        self.sub.on_connect = self.__on_connect
        self.sub.on_message = self.__on_message
        self.sub.connect(MQTT_SERVER, 1883, 60)

        s = threading.Thread(target=self.__sub, args=(self.sub,))
        s.daemon = True
        s.start()

        self.pub = mqtt.Client(client_id="motor_power_pub", transport="TCP")
        self.pub.connect(MQTT_SERVER, 1883, 60)

        self.steps = 0
        self.done = False
        self.reward = 0
        self.rewards = []
        self.info = None
        self.isResetComplete=False
        self_env = self

    @staticmethod
    def __on_connect(client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc))
        client.subscribe(topic=MQTT_PENDULUM_ANGLE_TOPIC)
        client.subscribe(topic=MQTT_MOTOR_ANGLE_TOPIC)

    @staticmethod
    def __on_message(client, userdata, msg):
        if msg.topic == MQTT_PENDULUM_ANGLE_TOPIC:
            pendulum_info = str(msg.payload)
            msg = pendulum_info.split(":")
            theta = int(msg[1].split(',')[0])
            rpm = int(msg[2].split("'")[0])
            self_env.__insert_angle_info_to_buffer(device=Device.pendulum, theta=theta, rpm=rpm)
            #print("Pendulum Angle value: {0}, {1}".format(angle, speed))
        elif msg.topic == MQTT_MOTOR_ANGLE_TOPIC:
            motor_info = str(msg.payload)
            msg = motor_info.split(":")
            theta = int(msg[1].split("'")[0])
            self_env.__insert_angle_info_to_buffer(device=Device.motor, theta=theta, rpm=0)
            #print("Motor Angle value: {0}, {1}".format(angle, speed))
        elif True: # Reset Complete Test
            pass

    def __insert_angle_info_to_buffer(self, device, theta, rpm):
        rad = math.radians(theta)
        cosine_theta = math.cos(float(rad))
        sine_theta = math.sin(float(rad))
        time_epoch = datetime.now().timestamp()
        if device == Device.pendulum:
            # normalization, rpm / 20
            self.pendulum_info_buffer.append((time_epoch, rad, [cosine_theta, sine_theta, rpm / 20]))
        elif device == Device.motor:
            self.motor_info_buffer.append((time_epoch, rad, [cosine_theta, sine_theta]))
        else:
            raise AttributeError("Unrecognized Device")

    @staticmethod
    def __sub(sub):
        try:
            print("Sub thread started!")
            sub.loop_forever()
        except KeyboardInterrupt:
            print("Sub Interrupted!")
            sub.unsubscribe([MQTT_PENDULUM_ANGLE_TOPIC, MQTT_MOTOR_ANGLE_TOPIC])


    ################################################
    ###  Environment Methods: reset, step, close ###
    ################################################
    def reset(self):
        # reset position
        self.steps = 0
        self.reward = 0
        self.done = False
        del self.rewards[:]

        # reset position
        self.pub.publish(topic=MQTT_MOTOR_POWER_TOPIC, payload="speed:" + str(999999))

        while not self.isResetComplete:
            time.sleep(0.05)

        self.pub.publish(topic=MQTT_MOTOR_POWER_TOPIC, payload="rectify")

        time.sleep(7.0)

        self.pendulum_info_buffer.clear()
        self.motor_info_buffer.clear()

        # wait while buffers are not empty
        while True:
            if len(self.pendulum_info_buffer) != 0 and len(self.motor_info_buffer) != 0:
                break
            # else:
            #     print("{0} - {1}".format(len(self.pendulum_info_buffer), len(self.motor_info_buffer)))
            time.sleep(0.001)

        # pendulum_info = [time_epoch, radian, [cos(radian), sin(radian), rpm]]
        pendulum_info = self.pendulum_info_buffer.pop()
        # motor_info = [time_epoch, theta, [cos(radian), sin(radian)]]
        motor_info = self.motor_info_buffer.pop()


        # return [cos(p_rad), sin([p_rad), p_rpm, cos(m_rad), sin(m_rad)]
        return pendulum_info[2] + motor_info[2]

    def step(self, action_index):
        motor_speed = (action_index - 5) * 20
        self.steps += 1

        # set action
        self.pub.publish(topic=MQTT_MOTOR_POWER_TOPIC, payload="speed:" + str(motor_speed))

        # wait while buffers are not empty
        while True:
            if len(self.pendulum_info_buffer) != 0 and len(self.motor_info_buffer) != 0:
                break
            time.sleep(0.001)
            # print("step:  {0}  {1}".format(len(self.pendulum_info_buffer), len(self.motor_info_buffer)))

        # pendulum_info = [time_epoch, radian, [cos(radian), sin(radian), rpm]]
        pendulum_info = self.pendulum_info_buffer.pop()
        # motor_info = [time_epoch, radian, [cos(radian), sin(radian)]]
        motor_info = self.motor_info_buffer.pop()

        # action normalization (-2 ~ 2)
        n_action = motor_speed / 50.0
        # reward = -(pendulum_radian^2 + 0.1 * pendulum_theta_dot^2 + 0.01 * action^2)
        self.reward = -((pendulum_info[1] ** 2) + (0.1 * (pendulum_info[2][2] ** 2)) + (0.01 * (n_action ** 2)))
        self.reward += 1 if pendulum_info[1] ** 2 < 0.1 else 0
        self.rewards.append(self.reward)

        self.isDone(pendulum_info[1], motor_info[1])

        # pendulum_angle_time_epoch = datetime.fromtimestamp(
        #     pendulum_angle[0]
        # ).strftime('%Y-%m-%d %H:%M:%S')

        # motor_angle_time_epoch = datetime.fromtimestamp(
        #     motor_angle[0]
        # ).strftime('%Y-%m-%d %H:%M:%S')

        # print("Inside step: pendulum_angle_time_epoch - {0}, motor_angle_time_epoch - {1}". format(
        #     pendulum_angle_time_epoch,
        #     motor_angle_time_epoch
        # ))

        # return state, reward, done, info
        return pendulum_info[2] + motor_info[2], self.reward, self.done, self.info

    def episode_reset(self):
        self.steps = 0
        self.reward = 0
        self.done = False
        del self.rewards[:]

        # reset position
        self.pub.publish(topic=MQTT_MOTOR_POWER_TOPIC, payload="speed:" + str(999999))
        self.pendulum_info_buffer.clear()
        self.motor_info_buffer.clear()

    def close(self):
        self.pub.publish(topic=MQTT_MOTOR_POWER_TOPIC, payload="speed:0")   # set motor speed 0

    def isDone(self, pendulum_angle, motor_pos):
        # if pendulum_angle < -0.436 or pendulum_angle > 0.436:
        if pendulum_angle < -1.57 or pendulum_angle > 1.57:
            self.info = "fall down: " + str(pendulum_angle)
            self.done = True
        elif self.steps > 300:                        # maximum step
            self.info = "max step"
            self.done = True
        elif np.mean(self.rewards[-30:]) > 20:      # success
            self.info = "success"
            self.done = True
        elif motor_pos > 3.141592 * 2 or motor_pos < -3.141592 * 2:   # limit motor position
            self.info = "limit position, pos:" + str(round(motor_pos, 4))
            self.reward = -100
            self.done = True
        else:
            self.done = False
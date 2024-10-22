import paho.mqtt.client as mqtt
import numpy as np
import threading
import math
import time
from datetime import datetime
import random
import json
import json
import urllib
import os


MQTT_SERVER = '10.0.0.1'

MQTT_PUB_TO_SERVO_POWER = 'motor_power'
MQTT_PUB_RESET = 'reset'
MQTT_SUB_FROM_SERVO = 'servo_info'
MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info'
MQTT_SUB_RESET_COMPLETE = 'reset_complete'
MQTT_ERROR = 'error'

STATE_SIZE = 4
MAX_BUFFER_SIZE = 5

balance_motor_power_list = [-60, 0, 60]
# balance_motor_power_list = [-350, -300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300, 350]
# balance_motor_power_list = [-350, -200, -120, -70, -30, 0, 30, 70, 120, 200, 350]
# balance_motor_power_list = [-250, -170, -100, -90, -40, 0, 40, 90, 100, 170, 250]
# swingup_motor_power_list = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]

self_env = None
PUB_ID = 0

class Env:
    def __init__(self):
        global self_env

        self_env = self

        self.episode = 0

        self.state_space_shape = (STATE_SIZE,)
        self.action_space_shape = (len(balance_motor_power_list),)

        self.reward = 0

        self.steps = 0
        self.pendulum_radians = []
        self.current_state = []
        self.current_pendulum_radian = 0
        self.current_pendulum_velocity = 0
        self.current_motor_velocity = 0

        self.theta_n_k1 = 0
        self.theta_dot_k1 = 0
        self.alpha_n_k1 = 0
        self.alpha_dot_k1 = 0

        self.is_swing_up = True
        self.is_state_changed = False
        self.is_motor_limit = False
        self.is_limit_complete = False
        self.is_reset_complete = False

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

    @staticmethod
    def __on_connect(client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc), flush=False)
        client.subscribe(topic=MQTT_SUB_FROM_SERVO)
        client.subscribe(topic=MQTT_SUB_MOTOR_LIMIT)
        client.subscribe(topic=MQTT_SUB_RESET_COMPLETE)
        client.subscribe(topic=MQTT_ERROR)

    @staticmethod
    def __sub(sub):
        try:
            print("***** Sub thread started!!! *****", flush=False)
            sub.loop_forever()
        except KeyboardInterrupt:
            print("Sub thread KeyboardInterrupted", flush=False)
            sub.unsubscribe(MQTT_SUB_FROM_SERVO)
            sub.unsubscribe(MQTT_SUB_MOTOR_LIMIT)
            sub.unsubscribe(MQTT_SUB_RESET_COMPLETE)
            sub.unsubscribe(MQTT_ERROR)
            sub.disconnect()

    def __pub(self, topic, payload, require_response=True):
        global PUB_ID
        self.pub.publish(topic=topic, payload=payload)
        # print("<=== pub:", datetime.utcnow().strftime('%S.%f')[:-1], "-", PUB_ID, payload, flush=False)

        PUB_ID += 1

        if require_response:
            is_sub = False
            while not is_sub:
                if self.is_state_changed or self.is_limit_complete or self.is_reset_complete:
                    is_sub = True
                time.sleep(0.0001)
        # while not self.is_state_changed:
        #     time.sleep(0.0001)
        # while not self.is_limit_complete:
        #     time.sleep(0.0001)
        # while not self.is_reset_complete:
        #     time.sleep(0.0001)

        self.is_state_changed = False
        self.is_limit_complete = False
        self.is_reset_complete = False

    @staticmethod
    def __on_message(client, userdata, msg):
        global PUB_ID

        if msg.topic == MQTT_ERROR:
            os.system('/home/link/anaconda3/bin/python3.6 PushSlack.py')

        elif msg.topic == MQTT_SUB_FROM_SERVO:

            #servo_info = str(msg.payload.decode("utf-8"))
            servo_info = json.loads(msg.payload.decode("utf-8"))
            motor_radian = float(servo_info["motor_radian"])
            motor_velocity = float(servo_info["motor_velocity"])
            pendulum_radian = float(servo_info["pendulum_radian"])
            pendulum_velocity = float(servo_info["pendulum_velocity"])
            pub_id = servo_info["pub_id"]

            # print("---> sub:", datetime.utcnow().strftime('%S.%f')[:-1], "- pub_id:", pub_id, flush=False)

            # if abs(pendulum_radian) < 3.14 / 24:
            #     self_env.is_swing_up = False

            self_env.__set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

        elif msg.topic == MQTT_SUB_MOTOR_LIMIT:
            info = str(msg.payload.decode("utf-8")).split('|')
            pub_id = info[1]
            if info[0] == "limit_position":
                self_env.is_motor_limit = True
            elif info[0] == "reset_complete":
                self_env.is_limit_complete = True

        elif msg.topic == MQTT_SUB_RESET_COMPLETE:
            self_env.is_reset_complete = True
            servo_info = str(msg.payload.decode("utf-8")).split('|')
            motor_radian = float(servo_info[0])
            motor_velocity = float(servo_info[1])
            pendulum_radian = float(servo_info[2])
            pendulum_velocity = float(servo_info[3])
            pub_id = servo_info[4]
            self_env.theta_n_k1 = float(servo_info[5])
            self_env.theta_dot_k1 = float(servo_info[6])
            self_env.alpha_n_k1 = float(servo_info[7])
            self_env.alpha_dot_k1 = float(servo_info[8])
            self_env.__set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

    def __set_state(self, motor_radian, motor_velocity, pendulum_radian, pendulum_velocity):
        self.is_state_changed = True
        # motor_cosine_theta = math.cos(motor_radian)
        # motor_sine_theta = math.sin(motor_radian)
        # pendulum_cosine_theta = math.cos(pendulum_radian)
        # pendulum_sine_theta = math.sin(pendulum_radian)

        self.current_state = [pendulum_radian, pendulum_velocity, motor_radian, motor_velocity]
            #[pendulum_cosine_theta, pendulum_sine_theta, pendulum_velocity]
        # motor_cosine_theta, motor_sine_theta, motor_velocity

        self.current_pendulum_radian = pendulum_radian
        self.current_pendulum_velocity = pendulum_velocity
        self.current_motor_velocity = motor_velocity

    def __pendulum_reset(self):
        self.__pub(
            MQTT_PUB_TO_SERVO_POWER,
            "0|pendulum_reset|{0}".format(PUB_ID),
            require_response=False
        )

    # def __swing_up_by_manual(self):
    #     while self.is_swing_up:
    #         pendulum_radian = self.current_pendulum_radian
    #         pendulum_velocity = self.current_pendulum_velocity
    #
    #         # servo_loop
    #         voltage = 82
    #         if abs(pendulum_velocity) > 40:
    #             voltage /= int(0.5 * np.log(abs(pendulum_velocity)))
    #
    #         # timer
    #         # voltage = 32
    #         # if abs(pendulum_velocity) > 20:
    #         #     voltage /= int(1.3 * np.log(abs(pendulum_velocity)))
    #
    #         if abs(pendulum_radian) < math.pi / 2:
    #             voltage /= 2
    #
    #         if pendulum_radian >= 0:
    #             pendulum_radian = math.pi - pendulum_radian
    #         else:
    #             pendulum_radian = - math.pi + abs(pendulum_radian)
    #
    #         if pendulum_veloc1ity < 0:
    #             motor_power = int(-2*math.cos(pendulum_radian) * voltage)
    #         else:
    #             motor_power = int(2*math.cos(pendulum_radian) * voltage)
    #         self.__pub(MQTT_PUB_TO_SERVO_POWER, "{0}|swingup|{1}".format(motor_power, PUB_ID))
    #         self.is_motor_limit = False

    def manual_swingup_balance(self):
        self.__pub(MQTT_PUB_RESET, "reset|{0}".format(PUB_ID))

    def wait(self):
        self.__pub(MQTT_PUB_TO_SERVO_POWER, "0|wait|{0}".format(PUB_ID))

    def reset(self):
        self.steps = 0
        self.pendulum_radians = []
        self.reward = 0
        self.is_motor_limit = False

        wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3

        previousTime = time.perf_counter()
        time_done = False

        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= wait_time:
                time_done = True
            time.sleep(0.0001)

        # if self.episode % 10 == 0:
        self.__pendulum_reset()

        self.wait()

        # self.is_swing_up = True
        # self.__swing_up_by_manual()
        # print("***** Swing up complete *****", flush=False)
        self.manual_swingup_balance()
        self.is_motor_limit = False

        self.episode += 1

        return self.current_state, self.theta_n_k1, self.theta_dot_k1, self.alpha_n_k1, self.alpha_dot_k1

    def step(self, action_index):
        motor_power = balance_motor_power_list[action_index]

        self.__pub(MQTT_PUB_TO_SERVO_POWER, "{0}|{1}|{2}".format(motor_power, "balance", PUB_ID))


        pendulum_radian = self.current_pendulum_radian
        pendulum_angular_velocity = self.current_pendulum_velocity

        # self.reward = 1# if abs(pendulum_radian) > 0.02 else 5

        # self.reward = 1 / (abs(pendulum_radian) + 0.001)

        self.reward = 1
        # self.reward = 1 - (100 * (pendulum_radian ** 2))
        # self.reward = 1 - (pendulum_radian ** 2 + 0.1 * (pendulum_angular_velocity/2 ** 2) + \
        #               0.001 * (np.clip(motor_power, -2., 2.)**2))

        self.steps += 1
        self.pendulum_radians.append(pendulum_radian)
        done, info = self.__isDone()

        return self.current_state, self.reward, done, info

    def __isDone(self):
        if self.steps >= 5000:# len(self.pendulum_radians) > 500 and abs(np.mean(self.pendulum_radians[-500:])) < 3.14 / 18:
            return True, "*** Success!!! ***"
        elif self.is_motor_limit:
            self.reward = -100
            return True, "*** Limit position ***"
        elif abs(self.pendulum_radians[-1]) > 3.14 / 24:
            self.is_fail = True
            self.reward = -100
            return True, "*** Fail!!! ***"
        else:
            return False, ""

    def close(self):
        print("*************** Close ***************", flush=False)
        self.pub.publish(topic=MQTT_PUB_TO_SERVO_POWER, payload=str(0))


# if __name__ == "__main__":
#     try:
#         env = Env(False)
#         # env.reset()
#
#         count = 0
#         for i in range(1000000):
#             env.reset()
#             done = False
#             while not done:
#                 state, reward, done, info, is_swing_up = env.step(5)
#                 print("|| state: {0:2f} || reward: {1} || done: {2} || info: {3} || time: {4} ||".format(
#                     math.acos(state[0]), reward, done, info, datetime.utcnow().strftime('%S.%f')[:-3]
#                 ))
#                 time.sleep(0.001)
#         env.close()
#     except KeyboardInterrupt as e:
#         env.close()

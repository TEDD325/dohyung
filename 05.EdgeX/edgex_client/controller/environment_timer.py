import threading
import math
from collections import deque
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np
import json
import requests

MQTT_SERVER = '192.168.137.10'

MQTT_MOTOR_POWER = 'motor_power'

MQTT_STATE = 'state_info_from_edgex'
MQTT_MOTOR_POWER_RESPONSE = 'motor_power_response_from_edgex'

sub_topic_list = [MQTT_STATE]

motor_speed_list = [-100, -70, -50, -30, -10, 0, 10, 30, 50, 70, 100]
# motor_speed_list = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]

self_env = None

motor_power_msg_old = {'name': 'raspi-mqtt-motor-device', 'motor_power': "10000000"}

motor_power_msg = {'motor_power': "10000000"}
motor_reponse_msg = {'motor_status':"1"}
headers = {"Content-Type" : "application/json"}

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
        #self.isResetComplete = False
        self.reward = 0
        self.rewards = []
        self.info = None

    @staticmethod
    def __on_connect(client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc))
        client.subscribe(topic=MQTT_STATE)
        client.subscribe(topic=MQTT_MOTOR_POWER_RESPONSE)

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
        #print(msg.topic + ' ' + ': {0}'.format(datetime.utcnow().strftime('%H-%M-%S.%f')[:-3]))
        m_decode = str(msg.payload.decode("utf-8"))
        if msg.topic == MQTT_STATE:
            state_info = json.loads(m_decode)
            if 'pendulum_angle' == state_info['readings'][0]['name']:
                pendulum_info = state_info['readings'][0]['value']
                self_env.__insert_angle_info_to_buffer('pendulum', pendulum_info) #pendulum_info is only angle value
            elif 'motor_angle' == state_info['readings'][0]['name']:
                motor_info = state_info['readings'][0]['value']
                self_env.__insert_angle_info_to_buffer('motor', motor_info)

        if msg.topic == MQTT_MOTOR_POWER_RESPONSE:
            motor_power_response = json.loads(m_decode)
            print(motor_power_response + "!!!!") #debug
            if motor_power_response['readings'][0]['value'] == "0":
                # self_env.isResetComplete = True
                self_env.reset() # reset 함수 호출
            if motor_power_response['readings'][0]['value'] == "1":
                pass

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

            motor_cosine_theta = math.cos(float(motor_radian))
            motor_sine_theta = math.sin(float(motor_radian))

            motor_angular_velocity = (motor_radian - self.last_motor_radian) / 0.015
            self.last_motor_radian = motor_radian

            self.motor_info_list.append([motor_radian,
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
        self.isResetComplete = True
        del self.rewards[:]
        del self.state_list[:]

        # reset position
        # motor_power_reset_msg_json = json.dumps(motor_power_msg)
        # self.pub.publish(topic=MQTT_MOTOR_POWER, payload=motor_power_reset_msg_json)
        start_time = time.time()
        requests.put(
            #url="http://localhost:49982/api/v1/device/5b97b4ea44d07f79eaccf731/motor_status",
            url="http://localhost:48082/api/v1/device/5b97f24e44d00aaaf3e8d413/command/5b97f11544d00aaaf3e8d40e",
            data=json.dumps(motor_reponse_msg),
            headers=headers
        )
        end_time = time.time()
        print("[!!!!]TIME: ", end_time - start_time)
        print("Reset Msg Sent!")

        time.sleep(0.05)

        # pendulum_info_list = [radian, time, [cos(radian), sin(radian), angular_velocity]]
        self.pendulum_state = self.pendulum_info_list
        # motor_info_list = [radian, time, [cos(radian), sin(radian)]]
        self.motor_state = self.motor_info_list

        # reshape state (total 30 values)
        for i in range(self.list_size):
            print(self.pendulum_state)
            print(self.motor_state)
            self.state_list += self.pendulum_state[i][1]
            self.state_list += self.motor_state[i][1]

        return self.state_list

    def step(self, action_index):
        self.steps += 1

        del self.state_list[:]

        # motor_speed = (action_index - 5) * 20
        motor_speed = motor_speed_list[action_index]

        # set action

        # motor_power_msg["motor_power"] = str(motor_speed)
        # motor_power_msg_json = json.dumps(motor_power_msg)
        # self.pub.publish(topic=MQTT_MOTOR_POWER, payload=motor_power_msg_json)

        start_time = time.time()
        motor_power_msg["motor_power"] = str(motor_speed)
        requests.put(
            #url="http://localhost:49982/api/v1/device/5b97b4ea44d07f79eaccf731/motor_power",
            url="http://localhost:48082/api/v1/device/5b97f24e44d00aaaf3e8d413/command/5b97f11544d00aaaf3e8d40d",
            data=json.dumps(motor_power_msg),
            headers=headers
        )
        end_time = time.time()
        print("[!!!!]TIME: ", end_time - start_time)

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
            # self.pub.publish(topic=MQTT_MOTOR_POWER, payload=str(0))
            requests.put(
                #url="http://localhost:49982/api/v1/device/5b97b4ea44d07f79eaccf731/motor_power",
                url="http://localhost:48082/api/v1/device/5b97f24e44d00aaaf3e8d413/command/5b97f11544d00aaaf3e8d40d",
                data=json.dumps(motor_power_msg),
                headers=headers
            )

        # reshape state (total 30 values)
        for i in range(self.list_size):
            self.state_list += self.pendulum_state[i][1]
            self.state_list += self.motor_state[i][1]

        # pendulum angular velocity, motor angular velocity
        # print(self.pendulum_state[4][2][2], self.motor_state[4][2][2])

        # return state, reward, done, info
        return self.state_list, self.reward, self.done, self.info

    def close(self):
        motor_power_reset_msg_json = json.dumps(motor_power_msg)
        # self.pub.publish(topic=MQTT_MOTOR_POWER, payload=motor_power_reset_msg_json)
        #
        requests.put(
            #url="http://localhost:49982/api/v1/device/5b97b4ea44d07f79eaccf731/motor_power",
            url="http://localhost:48082/api/v1/device/5b97f24e44d00aaaf3e8d413/command/5b97f11544d00aaaf3e8d40d",
            data=json.dumps(motor_power_msg),
            headers=headers
        )

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
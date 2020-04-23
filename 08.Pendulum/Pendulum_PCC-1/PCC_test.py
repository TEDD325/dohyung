import spidev
import time
import math
import pickle
import pickle
# import pandas as pd
import numpy as np
import sys

self_servo = None

class QubeServo2:
    def __init__(self):
        global self_servo
        self_servo = self

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.mode = 0b10
        self.spi.max_speed_hz = 1000000

    def data_conversion(self, data):
        # Devoid ID
        device_id = ((data[0] & 0xff) << 8) | (data[1] & 0xff)

        # Motor Encoder Counts
        encoder0 = ((data[2] & 0xff) << 16) | ((data[3] & 0xff) << 8) | (data[4] & 0xff)
        if encoder0 & 0x00800000:
            encoder0 = encoder0 | 0xFF000000
            # 2's complement calculate
            encoder0 = (0x100000000 - encoder0) * (-1)

        # convert the arm encoder counts to angle theta in radians
        motor_radian = encoder0 * (-2.0 * math.pi / 2048.0)
        motor_angle = motor_radian * 57.295779513082320876798154814105

        # Pendulum Encoder Counts
        encoder1 = ((data[5] & 0xff) << 16) | ((data[6] & 0xff) << 8) | (data[7] & 0xff)
        if encoder1 & 0x00800000:
            encoder1 = encoder1 | 0xFF000000
            # 2's complement calculate
            encoder1 = (0x100000000 - encoder1) * (-1)

        # wrap the pendulum encoder counts when the pendulum is rotated more than 360 degrees
        encoder1 = encoder1 % 2048
        if encoder1 < 0:
            encoder1 += 2048

        # convert the arm encoder counts to angle theta in radians
        pendulum_radian = encoder1 * (2.0 * math.pi / 2048.0) - math.pi
        pendulum_angle = pendulum_radian * 57.295779513082320876798154814105

        return device_id, motor_angle, motor_radian, pendulum_angle, pendulum_radian

    def reset(self):
        self.spi.xfer2([
            0x01,
            0x00,
            0b01111111,
            # LED Magenta
            0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])
        data = self.spi.xfer2([
            0x01,
            0x00,
            0b01111111,
            # LED Magenta
            0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])
        device_id, motor_angle, motor_radian, pendulum_angle, pendulum_radian = self.data_conversion(data)
        print("\n\n*--*--*--*--* Reset Complete --> || Device ID:{0} || "
              "Motor Angle:{1:3.1f} || Pendulum Angle:{2:3.1f} || *--*--*--*--*\n"
              .format(device_id, motor_angle, pendulum_angle))

    def __motor_command_split(self, motor_command):
        # to signed
        if motor_command & 0x0400:
            motor_command = motor_command | 0xfc00

        # add amplifier bit
        motor_command = (motor_command & 0x7fff) | 0x8000

        # separate into 2 bytes
        motor_command_h = (motor_command & 0xff00) >> 8
        motor_command_l = (motor_command & 0xff)
        return motor_command_h, motor_command_l

    def read_data(self):
        data = self.spi.xfer2([
            0x01,
            0x00,
            0x1f,
            0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])
        device_id, motor_angle, motor_radian, pendulum_angle, pendulum_radian = self.data_conversion(data)
        return motor_radian, pendulum_radian

    def __set_motor_command(self, motor_command, color):
        if color == "red":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x00, 0x00, 0x00, 0x00
        elif color == "green":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x03, 0xe7, 0x00, 0x00
        elif color == "blue":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x00, 0x00, 0x03, 0xe7
        elif color == "cyan":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x03, 0xe7, 0x03, 0xe7
        elif color == "magenta":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7
        elif color == "yellow":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x03, 0xe7, 0x00, 0x00
        elif color == "white":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x03, 0xe7, 0x03, 0xe7
        else:
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

        motor_command_h, motor_command_l = self.__motor_command_split(motor_command)

        data = self.spi.xfer2([
            0x01,
            0x00,
            0x1f,
            red_h, red_l, green_h, green_l, blue_h, blue_l,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            motor_command_h, motor_command_l
        ])
        device_id, motor_angle, motor_radian, pendulum_angle, pendulum_radian = self.data_conversion(data)

        return motor_radian, pendulum_radian

    def get_pcc(self, filename_1, filename_2):
        data_1 = self.loadPickle(filename_1)
        data_2 = self.loadPickle(filename_2)

        len_data_pen_1 = len(data_1)
        len_data_pen_2 = len(data_2)
        # print(len_data_pen_1)
        # print(len_data_pen_2)

        if len_data_pen_1 != len_data_pen_2:
            min_len = min(len_data_pen_1, len_data_pen_2)
            data_pen_1 = data_1[:min_len]
            data_pen_2 = data_2[:min_len]
        # print(len(data_pen_1))
        # print(len(data_pen_2))

        np_data_pen_1 = np.array(data_1)
        np_data_pen_2 = np.array(data_2)
        # print(np_data_pen_1[:,:,1,:][:,-1,-1])
        # print(np_data_pen_1[:,:,1,:].shape)
        # print(np_data_pen_1)
        # print(np_data_pen_1)
        # print(len(np_data_pen_1[:,:,1,:][:,-1,-1]))

        # lst = []
        # lst.append(np_data_pen_1[:, :, col_num, :][:, -1, -1])  # 2: pendulum radian
        # lst.append(np_data_pen_2[:, :, col_num, :][:, -1, -1])

        # lst = []
        # lst.append(np_data_pen_1)  # 2: pendulum radian
        # lst.append(np_data_pen_2)

        # df = pd.DataFrame(lst).T
        # print(df)
        # corr = df.corr(method='pearson')
        # print("[pandas] corr:", corr)
        # print("np_data_pen_1:", np_data_pen_1, end="\n\n")
        # print("np_data_pen_2:", np_data_pen_2, end="\n\n")

        corr = np.corrcoef(np_data_pen_1, np_data_pen_2)
        print(corr)

    def run(self, motorPWM):
        motorPWM = abs(motorPWM)
        motor_radian_list = []
        pendulum_radian_list = []

        for i in range(1000):
            if i % 2 == 0:
                motor_radian, pendulum_radian = self.__set_motor_command(motorPWM, "blue")
            else:
                motor_radian, pendulum_radian = self.__set_motor_command(-motorPWM, "blue")
            motor_radian_list.append(motor_radian)
            pendulum_radian_list.append(pendulum_radian)
            print("motor_radian:", motor_radian)
            print("pendulum_radian: ", pendulum_radian)
            time.sleep(0.1)
        self.__set_motor_command(0, "blue")
        return motor_radian_list, pendulum_radian_list

    def savePickle(self, filename, state_list):
        with open(filename, 'wb') as f:
            pickle.dump(state_list, f, pickle.HIGHEST_PROTOCOL)

    def loadPickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == "__main__":
    servo = QubeServo2()
    machine = sys.argv[1]
    number = sys.argv[2]
    condition = sys.argv[3]

    if condition == 'measurement':
        filename_1 = 'Rpi_'+machine+'_['+number+']_motor_radian.pickle'
        filename_2 = 'Rpi_'+machine+'_['+number+']_pendulum_radian.pickle'
        motorPWM_abs = 60

        servo.reset()
        time.sleep(0.1)

        # measure state
        motor_radian_list, pendulum_radian_list = servo.run(motorPWM_abs)

        # save using pickle
        servo.savePickle(filename_1, motor_radian_list)
        servo.savePickle(filename_2, pendulum_radian_list)

        time.sleep(1)
        servo.reset()
        time.sleep(10)

    elif condition == 'pcc':

        filename_1_motor = 'Rpi_1_[1]_motor_radian.pickle'
        filename_2_motor = 'Rpi_1_[2]_motor_radian.pickle'
        filename_1_pendulum = 'Rpi_1_[1]_pendulum_radian.pickle'
        filename_2_pendulum = 'Rpi_1_[2]_pendulum_radian.pickle'
        print("corr of motor_radian")
        servo.get_pcc(filename_1_motor, filename_2_motor)
        print()
        print("corr of pendulum_radian")
        servo.get_pcc(filename_1_pendulum, filename_2_pendulum)








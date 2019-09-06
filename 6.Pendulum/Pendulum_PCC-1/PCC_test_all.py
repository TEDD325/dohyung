import pickle
import numpy as np

class PCC:
    def __init__(self):
        pass

    def savePickle(self, filename, state_list):
        with open(filename, 'wb') as f:
            pickle.dump(state_list, f, pickle.HIGHEST_PROTOCOL)

    def loadPickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

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

if __name__ == "__main__":
    servo = PCC()

    print("------Rpi_1------")
    filename_1_motor = 'Rpi_1_[1]_motor_radian.pickle'
    filename_2_motor = 'Rpi_1_[2]_motor_radian.pickle'
    filename_1_pendulum = 'Rpi_1_[1]_pendulum_radian.pickle'
    filename_2_pendulum = 'Rpi_1_[2]_pendulum_radian.pickle'
    print("[corr of motor_radian]")
    servo.get_pcc(filename_1_motor, filename_2_motor)
    print("[corr of pendulum_radian]")
    servo.get_pcc(filename_1_pendulum, filename_2_pendulum)
    print()
    print('==============================')
    print()

    print("------Rpi_2------")
    filename_1_motor = 'Rpi_2_[1]_motor_radian.pickle'
    filename_2_motor = 'Rpi_2_[2]_motor_radian.pickle'
    filename_1_pendulum = 'Rpi_2_[1]_pendulum_radian.pickle'
    filename_2_pendulum = 'Rpi_2_[2]_pendulum_radian.pickle'
    print("[corr of motor_radian]")
    servo.get_pcc(filename_1_motor, filename_2_motor)
    print("[corr of pendulum_radian]")
    servo.get_pcc(filename_1_pendulum, filename_2_pendulum)
    print()
    print('==============================')
    print()

    print("------Rpi_1 & Rpi_2-first------")
    filename_1_motor = 'Rpi_1_[1]_motor_radian.pickle'
    filename_2_motor = 'Rpi_2_[1]_motor_radian.pickle'
    filename_1_pendulum = 'Rpi_1_[1]_pendulum_radian.pickle'
    filename_2_pendulum = 'Rpi_2_[1]_pendulum_radian.pickle'
    print("[corr of motor_radian]")
    servo.get_pcc(filename_1_motor, filename_2_motor)
    print("[corr of pendulum_radian]")
    servo.get_pcc(filename_1_pendulum, filename_2_pendulum)
    print()
    print("------Rpi_1 & Rpi_2-second------")
    filename_1_motor = 'Rpi_1_[2]_motor_radian.pickle'
    filename_2_motor = 'Rpi_2_[2]_motor_radian.pickle'
    filename_1_pendulum = 'Rpi_1_[2]_pendulum_radian.pickle'
    filename_2_pendulum = 'Rpi_2_[2]_pendulum_radian.pickle'
    print("[corr of motor_radian]")
    servo.get_pcc(filename_1_motor, filename_2_motor)
    print()
    print("[corr of pendulum_radian]")
    servo.get_pcc(filename_1_pendulum, filename_2_pendulum)



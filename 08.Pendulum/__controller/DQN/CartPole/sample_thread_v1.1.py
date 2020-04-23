import threading

THREAD_NUM = 2

class DQN:
    def __init__(self):
        pass

    def train(self):
        Solver = [DQN_solver(idx) for idx in range(THREAD_NUM)]
        for dqn_agent in Solver:
            dqn_agent.start()

class DQN_solver(threading.Thread):
    def __init__(self, idx):
        threading.Thread.__init__(self)
        self.thread_id = idx

    def run(self):
        print("hello world")

if __name__ == "__main__":
    dqn = DQN()
    dqn.train()
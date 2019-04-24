import tensorflow as tf
import threading

N_WORKER = 2

class Worker(object):
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def work(self):
        print("Hello")

if __name__ == "__main__":

    workers = [Worker(worker_id=i) for i in range(N_WORKER)]

    COORD = tf.train.Coordinator()
    """
    tf.train.Coordinator(): A coordinator for threads.
    This class implements a simple mechanism to coordinate the termination of a set of threads.
    """
    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()


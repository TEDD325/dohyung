import tensorflow as tf
import threading

N_WORKER = 2

class Worker(object):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.counter = 0

    def work(self):
        while not COORD.should_stop():
            self.counter += 1
            print("'", self.worker_id, "'", "- Hello, count: ", self.counter, COORD.should_stop() is ", COORD.should_stop()", end='\n\n')

            if self.counter >= 5:
                COORD.request_stop()
                print(self.worker_id, "'s COORD.should_stop() is", COORD.should_stop(), ", count: ", self.counter)
                break
        print(self.worker_id, "'s Coordinator.should_stop() returns True.")

if __name__ == "__main__":

    workers = [Worker(worker_id=i) for i in range(N_WORKER)]

    COORD = tf.train.Coordinator()

    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)

    COORD.join(threads=threads)
    '''
    threads: List of threading.Threads
    '''


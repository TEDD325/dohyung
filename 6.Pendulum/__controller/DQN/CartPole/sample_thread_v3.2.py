from __future__ import print_function
import tensorflow as tf
import threading
import time

N_THREAD = 10
# N_THREAD = 1

class queue(object):
    def __init__(self):
        # 세션 실행
        self.sess = tf.InteractiveSession()

        # capacity=100인 First In First Out Queue 생성
        self.gen_random_normal = tf.random_normal(shape=())
        self.queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
        self.enque = self.queue.enqueue(self.gen_random_normal)
        self.deque = self.queue.dequeue()

    # 요소 10개 enqueue
    def add(self, coord, i):
        while not coord.should_stop():
            self.sess.run(self.enque)
            if i == 11:
                coord.request_stop()
                break

    def print_result(self):
        print("queue.size():", self.sess.run(self.queue.size()))
        print("dequeue operation:", self.sess.run(self.deque))
        time.sleep(0.001)
        print("queue.size():", self.sess.run(self.queue.size()))
        print("dequeue operation:", self.sess.run(self.deque))
        time.sleep(0.001)
        print("queue.size():", self.sess.run(self.queue.size()))
        print("dequeue operation:", self.sess.run(self.deque))
        print("dequeue_many op:", self.sess.run(self.queue.dequeue_many(self.sess.run(self.queue.size()))))

if __name__ == "__main__":
    q = queue()

    start = time.time()

    coord = tf.train.Coordinator()

    # 쓰레드 10개 생성 후, 각 쓰레드가 parallel하게 add 함수 수행
    threads = [threading.Thread(target=(q.add), args=(coord, i)) for i in range(N_THREAD)]
    coord.join(threads)
    for thread in threads:
        thread.start()

    end = time.time()
    print("Elapsed time: ", end - start)

    q.print_result()

    coord.request_stop()
    coord.join(threads)
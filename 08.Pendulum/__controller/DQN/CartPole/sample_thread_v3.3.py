from __future__ import print_function
import tensorflow as tf
import threading
import time

N_THREAD = 10
# N_THREAD = 1
min_after_dequeue = 1

class queue(object):
    def __init__(self):
        # 세션 실행
        self.sess = tf.InteractiveSession()

        # capacity=100인 First In First Out Queue 생성
        self.gen_random_normal = tf.random_normal(shape=())
        self.queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32], min_after_dequeue=min_after_dequeue)
        '''
        The min_after_dequeue argument allows the caller to specify a minimum number of elements 
        that will remain in the queue after a dequeue or dequeue_many operation completes, 
        to ensure a minimum level of mixing of elements. 
        This invariant is maintained by blocking those operations until sufficient elements have been enqueued. 
        The min_after_dequeue argument is ignored after the queue has been closed.
        '''
        self.enque = self.queue.enqueue(self.gen_random_normal)
        self.deque = self.queue.dequeue()


    def print_result(self):
        print("queue.size():", self.sess.run(self.queue.size()))
        print("dequeue operation:", self.sess.run(self.deque))
        time.sleep(0.001)
        print("queue.size():", self.sess.run(self.queue.size()))
        print("dequeue operation:", self.sess.run(self.deque))
        time.sleep(0.001)
        print("queue.size():", self.sess.run(self.queue.size()))
        print("dequeue operation:", self.sess.run(self.deque))
        # print("dequeue_many op:", self.sess.run(self.queue.dequeue_many(self.sess.run(self.queue.size())-min_after_dequeue)))
        '''
        dequeue_many연산이 왜 먹히지 않는 걸까?
        '''

if __name__ == "__main__":
    q = queue()

    start = time.time()

    # 쓰레드 10개 생성 후, 각 쓰레드가 parallel하게 add 함수 수행
    qr = tf.train.QueueRunner(q.queue, [q.enque] * N_THREAD)
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(q.sess, coord=coord, start=True)

    end = time.time()
    print("Elapsed time: ", end - start)

    q.print_result()

    coord.request_stop()
    coord.join(enqueue_threads)
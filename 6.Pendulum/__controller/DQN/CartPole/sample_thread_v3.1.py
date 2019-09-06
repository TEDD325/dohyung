from __future__ import print_function
import tensorflow as tf
import threading
import time

# N_THREAD = 10
N_THREAD = 1

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
    def add(self):
        for _ in range(10):
            self.sess.run(self.enque)

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
    # 쓰레드 10개 생성 후, 각 쓰레드가 parallel하게 add 함수 수행
    threads = [threading.Thread(target=(q.add), args=()) for i in range(N_THREAD)]
    for thread in threads:
        thread.start()
    end = time.time()
    print("Elapsed time: ", end - start)
    '''
    N_THREAD = 10
        Elapsed time:  0.0029931068420410156
        queue.size(): 20
        dequeue operation: -0.57747364
        queue.size(): 99
        dequeue operation: 0.38039732
        queue.size(): 98
        dequeue operation: 0.533288
        dequeue_many op: [ 1.5988214  -0.93177146  0.07418784  0.3226504   0.5827279  -0.35475332
         -0.720033    0.23784268  0.5701678   0.2734187   0.69029206  0.20891759
         -0.45212218 -1.9610034  -1.3712933   0.40947515 -0.07539    -0.18229581
          0.2565371   0.37458345  0.27770144  1.4482644   0.63391083 -2.5089583
          0.2912862  -0.5779037   1.896998    0.66764647  1.2241212  -1.471258
         -1.8747721  -1.6725608  -1.2297735   0.06750647 -1.9383298   1.2660587
         -0.44437954 -0.2350661  -0.10704553  0.4084561  -0.85446423  0.18781795
         -0.94147855 -0.02283826 -0.2682895   0.09405711  1.1004019   0.78825545
         -2.0826354   0.39098558  1.1515522  -0.7243498  -0.99236965  1.3550209
         -0.29243928  2.362229   -0.73102003 -0.22900139 -1.2253342  -1.6532812
         -0.9996746  -0.13974562 -0.55690485  1.2308276  -2.0473156   1.2708298
         -1.2076416   0.7065796  -0.8398264  -1.2094074  -0.68444246 -1.4205154
          1.8711401  -0.65024656 -0.2240912  -0.01381483 -0.8336935   0.45838138
          2.5729136   2.5055766   0.6179792  -0.34448746  0.4695113  -0.47626674
          0.11635163 -0.02946486  0.3208067   1.9670511   1.9000998   0.21783365
          0.01151013  0.07674612 -0.39211714 -0.15002802  0.10688539  1.8046931
          2.0488024 ]

    N_THREAD = 1    
        Elapsed time: 0.0005261898040771484
        queue.size(): 1
        dequeue operation: -1.8315632
        queue.size(): 9
        dequeue operation: -0.5798997
        queue.size(): 8
        dequeue operation: 0.7013294
        dequeue_many op: [ 1.5405793  -2.6251647   0.29685017 -0.65843123 -2.1599066   0.06157193
          0.6982629 ]
    '''

    q.print_result()
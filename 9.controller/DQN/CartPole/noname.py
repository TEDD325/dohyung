import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

logits = tf.log([[10., 10.]])
samples = tf.random.categorical(logits=logits,
								num_samples=5)
samples = sess.run(samples)
print(samples)
print(type(samples))
print(samples.shape)

'''
logits = [  first batch: [unnormalized log-probabilities for first classes, second classes, ..., N classes],
			second batch: [...],
			... : [...]
		 ]
		 --> 2D Tensor with shape (batch_size, num_classes)
num_samples : 결과가 num_samples 개 만큼 나오도록 한다.
'''


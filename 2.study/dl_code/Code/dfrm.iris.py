#!/usr/bin/python

import tensorflow as tf
import pandas     as pd
import numpy      as np

#########################
#                       #
#  Data Initialization  #
#                       #
#########################

#
# Convert Data.Frame to numpy array
#

dfrm = pd.read_csv('data/iris.csv')
data = dfrm.to_csv( header = False, index = False ).split('\n')

feature = np.asarray( list ( map ( lambda x : [ float(y) for y in x.split(',')[:-1] ], data[:-1] ) ) )
level   = np.asarray( list ( map ( lambda x : 0 if x.split(',')[-1] == 'setosa' else 1 if x.split(',')[-1] == 'versicolor' else 2, data[:-1] ) ) )

num_of_data     = len( feature )
num_of_feature  = len( feature[0] )
num_of_class    = 3

#
# Convert result to Softmax type
#

level_list      = [ [ 0 for i in range( num_of_class ) ] for i in range( num_of_data ) ]

for i in range( num_of_data ): level_list[i][ level[i] ] = 1
result = np.asarray( level_list )

#########################
#                       #
#    TensorFlow Init    #
#                       #
#########################

#
# Declaration
#

x = tf.placeholder( tf.float32, [ None, num_of_feature ] )
W = tf.Variable( tf.zeros( [ num_of_feature, num_of_class ] ) )
b = tf.Variable( tf.zeros( [ num_of_class ] ) )

y  = tf.nn.softmax( tf.matmul( x, W ) + b )
y_ = tf.placeholder( tf.float32, [ None, num_of_class ] )

cross_entropy = tf.reduce_mean( -tf.reduce_sum( y_ * tf.log(y) , reduction_indices = [1] ) )
train_step = tf.train.GradientDescentOptimizer( 0.1 ).minimize( cross_entropy )

#
# Initialization
#

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run( init )

#########################
#                       #
#    Run TensorFlow     #
#                       #
#########################

num_of_step = 200

for i in range( num_of_step ):
    sess.run( train_step, feed_dict = { x: feature, y_: result } )

    correct_prediction = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_, 1 ) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ))

    print( i, ':', sess.run( accuracy, feed_dict = { x: feature, y_: result } ) )

#########################
#                       #
#     Print Result      #
#                       #
#########################

W_s = sess.run(W)
b_s = sess.run(b)

print('W :', W_s, type(W_s), len(W_s) )
print('b :', b_s, type(b_s), len(b_s) )



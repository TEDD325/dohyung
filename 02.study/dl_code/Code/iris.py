
import tensorflow as tf
import pandas     as pd
import numpy      as np

#########################
#                       #
#  Data Initialization  #
#                       #
#########################

num_of_step = 200

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


for i in range( num_of_step ):
    sess.run( train_step, feed_dict = { x: feature, y_: result } )

    correct_prediction = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_, 1 ) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ))

    print( '\r', i, ':', sess.run( accuracy, feed_dict = { x: feature, y_: result } ), end = '' )
    
print()

#########################
#                       #
#     Print Result      #
#                       #
#########################

W_s = sess.run(W)
b_s = sess.run(b)

#print('W :', W_s, type(W_s), len(W_s) )
#print('b :', b_s, type(b_s), len(b_s) )


#########################
#                       #
#     Verification      #
#                       #
#########################


Weight  = np.matrix( W_s )
Bias    = np.matrix( b_s )
Feature = np.matrix( feature )
Result = ( Feature * Weight + Bias ).tolist()

Class = list( map ( lambda x : 'setosa' if x[0] >= x[1] and x[0] >= x[2] else 'versicolor' if x[1] >= x[0] and x[1] >= x[2] else 'virginica', Result ) )
Original = list ( map ( lambda x : x.split(',')[-1], data[:-1] ) )

print( [ [ i, orig, clas ] for i, orig, clas in zip( range( min( len(Original), len(Class) ) ), Original, Class ) if orig != clas ] )
#print( '\n'.join( [ str(elem) for elem in [ [ i, orig, clas ] for i, orig, clas in zip( range( min( len(Original), len(Class) ) ), Original, Class ) ] ] ) )

#print ( min( len(Original), len(Class) ) )
#print ( [ range( min( len(Original), len(Class) ) )] )

#print('Weight :\n', Weight )
#print('Bias :\n', Bias )
#print('Feature :\n', Feature )
#print('Result :\n', Result, type(Result) )
#print('Class :\n', Class, type(Class) )
#print('Original:\n', Original, type(Original) )

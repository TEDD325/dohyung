# Keras Sequential Model Example
# June 27, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


# global constants and hyper-parameters
MY_EPOCH = 10
MY_BATCH = 16


    ####################
    # DATABASE SETTING #
    ####################


# create a random DB
np.set_printoptions(precision = 3, suppress=True)

# generates random floating point number in [0, 1]
data = np.random.random((1000, 100))

# generates random integer in [0, 5]
labels = np.random.randint(6, size = (1000, 1))

print('\n== DATABASE SHAPE INFO ==')
print('Input shape = ', data.shape)
print('Output shape = ', labels.shape)

# exit()

print("\nFirst input:")
print(data[0])
print("\nFirst output:")
print(labels[0])

# exit()

    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# keras sequential model
model = Sequential()
model.add(Dense(32, activation = 'sigmoid', input_dim = 100))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.save('chap1-1.h5')

# exit()

# model compilation setting
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', 
        metrics = ['accuracy'])


# model training and saving
model.fit(data, labels, epochs = MY_EPOCH, batch_size = MY_BATCH, 
        verbose = 1)
model.save('chap1-2.h5')

# exit()

    ####################
    # MODEL EVALUATION #
    ####################

# exit()

score = model.evaluate(data, labels, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


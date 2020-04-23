# Denoising with Keras Autoencoder
# July 16, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist


# global constants and hyper-parameters
np.random.seed(1337)
IMG_DIM = 28
MY_SAMPLE = 421
MY_SHAPE = (28, 28, 1)
MY_EPOCH = 10
MY_BATCH = 200
CNN_KERNEL = 3
VECTOR_DIM = 16
CNN_FILTERS = [32, 64]
OUT_DIR = "./output"
MODEL_DIR = "./model"


# create directories
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


    ####################
    # DATABASE SETTING #
    ####################


# load MNIST dataset
# note that we do not need labels (Y_train and Y_test)
# because we are building an autoencoder
(X_train, _), (X_test, _) = mnist.load_data()


# print shape information
def show_shape():
    print('\n== DB SHAPE INFO ==')
    print('X_train shape = ', X_train.shape)
    print('X_test shape = ', X_test.shape)
    print()
show_shape()

# exit()


# reshape to (28, 28, 1) and normalize input images
# -1 means it is an unknown dimension 
# and we want numpy to figure it out
X_train = np.reshape(X_train, [-1, IMG_DIM, IMG_DIM, 1])
X_test = np.reshape(X_test, [-1, IMG_DIM, IMG_DIM, 1])
# print(X_train[0])
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
show_shape()
# print(X_train[0])
# exit()

# print shape information
def show_sample(data):
    sample = data[MY_SAMPLE]
    sample = sample.reshape(IMG_DIM, IMG_DIM)
    plt.imshow(sample, cmap = 'gray')
    plt.show()
show_sample(X_train)

# generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std = 0.5
noise = np.random.normal(loc = 0.5, scale = 0.5, size = X_train.shape)
X_train_noisy = X_train + noise
noise = np.random.normal(loc = 0.5, scale = 0.5, size = X_test.shape)
X_test_noisy = X_test + noise
show_sample(X_train_noisy)
# exit()

# adding noise may exceed normalized pixel values>1.0 or <0.0
# clip pixel values >1.0 to 1.0 and <0.0 to 0.0
print('\n== SAMPLE IMAGE DATA (BEFORE CLIPPING) ==')
print(X_train_noisy[MY_SAMPLE].flatten())
# exit()
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
print('\n== SAMPLE IMAGE DATA (AFTER CLIPPING) ==')
print(X_train_noisy[MY_SAMPLE].flatten())


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


###########
# Encoder #
###########
inputs = Input(shape = MY_SHAPE)
x = inputs


# stacking of CNN layers using the sizes in CNN_FILTERS
# number of parameters in CNN layers
# formula: (kernel_size X kernel_size X channel + 1) X filters
# conv_1: (3 X 3 + 1) * 32 = 320
# conv_2: (3 X 3 X 32 + 1) X 64 = 18,496
# stride of 2 reduces the image size by half
for filters in CNN_FILTERS:
    x = Conv2D(filters = filters,
               kernel_size = CNN_KERNEL,
               strides = 2,
               activation = 'relu',
               padding = 'same')(x)


# generate the output vector
x = Flatten()(x)
vector_out = Dense(VECTOR_DIM)(x)


# instantiate encoder model
encoder = Model(inputs, vector_out)
encoder.summary()
# exit()

###########
# Decoder #
###########
vector_in = Input(shape = (VECTOR_DIM,))


# convert the vector input to 3-dimensional array to enter CNN
# use the shape (7, 7, 64) that was earlier saved
# from vector to suitable shape for transposed conv
x = Dense(7 * 7 * 64)(vector_in)
x = Reshape((7, 7, 64))(x)


# stacking of CNN layers using the sizes in CNN_FILTERS
# note that Conv2DTranspose increases the image size by 2
for filters in reversed(CNN_FILTERS):
    x = Conv2DTranspose(filters = filters,
                        kernel_size = CNN_KERNEL,
                        strides = 2,
                        activation = 'relu',
                        padding = 'same')(x)


# reconstruct the denoised result
# we just need one image
outputs = Conv2DTranspose(filters = 1,
                          kernel_size = CNN_KERNEL,
                          padding = 'same',
                          activation = 'sigmoid')(x)


# instantiate decoder model
decoder = Model(vector_in, outputs)
decoder.summary()


###############
# Autoencoder #
###############
autoencoder = Model(inputs, decoder(encoder(inputs)))
autoencoder.summary()
# exit()

# Mean Square Error (MSE) loss function, Adam optimizer
# note that accuracy is not used 
# because autoencoder is not a supervised learning
autoencoder.compile(loss = 'mse', optimizer = 'adam')


# train the autoencoder
# note that the input is noisy image
# and the label is the original image, NOT the correct number

# 한 번 학습 시킨 후, 그 다음 번에는 아래의 세 줄을 주석처리하고 다시 돌려본다.
'''
autoencoder.fit(X_train_noisy, X_train,
                validation_data = (X_test_noisy, X_test),
                epochs = MY_EPOCH, batch_size = MY_BATCH)
'''
autoencoder.save(os.path.join(MODEL_DIR, 'chap3.h5'))


    ####################
    # MODEL EVALUATION #
    ####################


# predict the autoencoder output from corrupted test images
denoised = autoencoder.predict(X_test_noisy)


# 3 set of images for the first 10 test data
# 1st rows - original images
# 2nd rows - images corrupted by noise
# 3rd rows - denoised images
Nout = 10
plt.figure(figsize = (20, 6))

for i in range(Nout):
    # show the input image
    ax = plt.subplot(3, Nout, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # show the encoded image using histogram
    ax = plt.subplot(3, Nout, i + 1 + Nout)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # show the decoded image
    ax = plt.subplot(3, Nout, i + 1 + Nout + Nout)
    plt.imshow(denoised[i].reshape(IMG_DIM, IMG_DIM))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(os.path.join(OUT_DIR, 'chap3.png'))


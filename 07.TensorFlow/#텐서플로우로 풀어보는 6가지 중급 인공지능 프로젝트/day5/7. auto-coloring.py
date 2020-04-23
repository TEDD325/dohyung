# Auto-coloring with Keras Unet
# July 18, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import matplotlib.pyplot as plt
import os
import numpy as np
from keras import models, backend
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import UpSampling2D, BatchNormalization
from keras.layers import  Concatenate, Activation
from keras import datasets, utils
from sklearn.preprocessing import minmax_scale
from keras.models import Model


# global constants and hyper-parameters
DIM = 32
RGB_CH = 3
GRAY_CH = 1
MY_EPOCH = 10
MY_BATCH = 200
MY_SPLIT = 0.2


# directories
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


# read CIFAR10 database
# Unet does not require the labels (Y_train and Y_test)
(X_train, _), (X_test, _) = datasets.cifar10.load_data()
print('\n== SHAPE INFO ==')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)


# data scaling
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# convert color to gray: easy (the opposite is hard)
# the shape changes from (32, 32, 3) to (32, 32, 1)
# we use a special formula to make it gray
# "..." is called python ellipsis
def RGB2Gray(X):
    R = X[..., 0:1]
    G = X[..., 1:2]
    B = X[..., 2:3]

    return 0.299 * R + 0.587 * G + 0.114 * B


# obtain gray-scaled images of the train and test sets
X_train_gray = RGB2Gray(X_train)
X_test_gray = RGB2Gray(X_test)

print('\n== SHAPE INFO FOR GRAY IMAGES ==')
print('X_train_gray:', X_train_gray.shape)
print('X_test_gray:', X_test_gray.shape)


# print sample color image
# dimension has to be 3 to plot color images
plt.imshow(X_train[0])
plt.show()


# print sample gray image
# dimension has to be 2 to plot gray-scale images
plt.imshow(X_train_gray[0].reshape(DIM, DIM))
plt.gray()
plt.show()
plt.clf()

exit()

    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# CNN for encoding in Unet
def conv_unet(x, ch_out, mp_flag):

    # mp_flag decides if we do max pooling or not
    # if used, the image size reduces by 1/4
    x = MaxPooling2D((2, 2), padding = 'same')(x) if mp_flag else x

    # first convolution with 3x3 filter
    x = Conv2D(ch_out, (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Dropout(0.05)(x)

    # second convolution with 3x3 filter
    x = Conv2D(ch_out, (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    return x


# CNN for decoding in Unet
def deconv_unet(x, ext, ch_out):
    
    # upscaling to increase the image size by 4x
    x = UpSampling2D((2, 2))(x)

    # concatenation makes this ANN a UNET
    # add non-neighboring bypass connections
    # between ext (= old layer) and x (= currunt layer) 
    # axis = 3 is for RGB channel
    x = Concatenate(axis = 3)([x, ext])
    print('Shape after concatenation:', x.shape)

    # first convolution with 3x3 filter
    x = Conv2D(ch_out, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # second convolution with 3x3 filter
    x = Conv2D(ch_out, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    return x


# construct the encoder part of Unet
def build_enc():
    c1 = conv_unet(original, 16, False)
    print('\n== SHAPE INFO DURING UNET CONSTRUCTION ==')          
    print('Input shape to Unet:', input_shape)
    print('Shape after the first CNN:', c1.shape)

    c2 = conv_unet(c1, 32, True)
    print('Shape after the second CNN:', c2.shape)

    encoded = conv_unet(c2, 64, True)
    print('Shape of the encoder output:', encoded.shape, '\n')

    return c1, c2, encoded


# construct the decoder part of Unet
def build_dec(c1, c2, encoded):
    # connect c2 layer as bypass
    x = deconv_unet(encoded, c2, 32)
    print('Shape after the first de-CNN:', x.shape)

    # connect c1 layer as bypass
    x = deconv_unet(x, c1, 16)
    print('Shape after the second de-CNN:', x.shape)


    # one more CNN layer to produce the final output
    # sigmoid activation, instead of tanh, is used 
    # 3-channel color image is produced
    decoded = Conv2D(RGB_CH, (3, 3), activation = 'sigmoid', 
            padding = 'same')(x)
    print('Shape of the decoder output:', decoded.shape)

    return decoded


# forming the overall Unet
input_shape = (DIM, DIM, GRAY_CH)
original = Input(shape = input_shape)
c1, c2, encoded = build_enc()
decoded = build_dec(c1, c2, encoded)
unet = Model(inputs = original, outputs = decoded)
unet.summary()


# model training
# the inputs are gray images, and the outputs color images
unet.compile(optimizer = 'adadelta', loss = 'mse')
history = unet.fit(X_train_gray, X_train,
                    epochs = MY_EPOCH,
                    batch_size = MY_BATCH,
                    shuffle = True,
                    validation_split = MY_SPLIT)

unet.save(os.path.join(MODEL_DIR, 'chap7.h5'))


    ####################
    # MODEL EVALUATION #
    ####################


# plot loss and validation curves
def plot_loss(history):
    history = history.history
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)

    # save the plot as PNG file
    plt.savefig(os.path.join(OUT_DIR, 'chap7-plot.png'))
    print('\n== LOSS PLOT SAVED ==')
    plt.clf()


# pick the first 10 images in the database
# show 3 kinds of images per data: 
# original (top), auto-coloring (middle), and ground truth (bottom)
def show_images(X_test_gray):

    # use the Unet model to predict color images
    # inputs are gray-scale images of the test set
    pred = unet.predict(X_test_gray)


    # show the fist 10 images in the database
    n = 10
    plt.figure(figsize = (20, 6))

    for i in range(n):
        # gray-scale image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test_gray[i].reshape(32, 32), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # auto-colored image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(pred[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # ground truth image
        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(X_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.savefig(os.path.join(OUT_DIR, 'chap7-sample.png'))
    print('\n== SAMPLE COLORING RESULTS SAVED ==')
    plt.clf()


plot_loss(history)
show_images(X_test_gray)

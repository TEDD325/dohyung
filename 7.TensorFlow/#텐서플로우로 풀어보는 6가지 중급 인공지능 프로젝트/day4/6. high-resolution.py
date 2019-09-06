# Image Resolution Enhancement with Keras GAN
# July 21, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import os, glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from scipy.misc import resize, imread


# global constants and hyper-parameters
MY_EPOCH = 100
MY_BATCH = 1
LOW_SHAPE = (64, 64, 3)
HIGH_SHAPE = (256, 256, 3)
DIS_ANS = (MY_BATCH, 16, 16, 1)

MY_MOM = 0.8
MY_ALPHA = 0.2
MY_RESIDUAL = 15
MY_OPT = Adam(0.0002, 0.5)


# 0 is no: we do training from scratch
# 1 is yes: we do prediction using the saved models
TRAINING_DONE = 0


# directories
DB_DIR = "./database/celeb/*.*"
OUT_DIR = "./output"
MODEL_DIR = "./model"
LOG_DIR = "./logs"


# create directories
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


    ####################
    # DATABASE SETTING #
    ####################


# pick some random images from the celebrity database
def sample_images(batch_size):

    # make a list of all images inside the data directory
    # then choose a random batch of images
    # the images in the database has 178 x 218 resolution
    all_images = glob.glob(DB_DIR)
    images_batch = np.random.choice(all_images, size = batch_size)

    low_img = []
    high_img = []
    for img in images_batch:
        # Get an ndarray of the current image
        # image shape is (218, 178, 3)
        new_img = imread(img, mode = 'RGB')
        new_img = new_img.astype(np.float32)


        # Resize the image from (218, 178, 3) to
        # high_shape = (256, 256, 3)
        # low_shape = (64, 64, 3)
        new_img_high = imresize(new_img, HIGH_SHAPE)
        new_img_low = imresize(new_img, LOW_SHAPE)


        # Do a random horizontal flip
        if np.random.random() < 0.5:
            new_img_high = np.fliplr(new_img_high)
            new_img_low = np.fliplr(new_img_low)


        high_img.append(new_img_high)
        low_img.append(new_img_low)

    # convert the lists to Numpy arrays
    return np.array(high_img), np.array(low_img)


    ##################
    # MODEL BUILDING #
    ##################


# use pre-trained VGG model
# it has 143 million pre-trained parameters
def build_vgg():

    # Load a pre-trained VGG19 model trained on 'Imagenet' dataset
    # need to decide which layer to use for the output
    input_shape = HIGH_SHAPE
    vgg = VGG19(weights = "imagenet")
    vgg.outputs = [vgg.layers[9].output]

    print('\n== VGG MODEL SUMMARY ==')
    print('Input shape:', input_shape)


    # extract features
    # shape is (?, 64, 64, 256)
    input_layer = Input(shape = input_shape)
    features = vgg(input_layer)
    print('Feature shape extracted:', features.shape)


    # create a Keras model
    # we do not train VGG19 further
    model = Model(inputs = [input_layer], outputs = [features])
    model.trainable = False
    model.summary()

    return model


# common convolution definition
def MY_CONV(channel, kernel, stride):
    return Conv2D(channel, kernel, strides = stride, padding = 'SAME')


# Create a discriminator network using deep CNN
# it has 5 million trainable parameters
def build_discriminator():

    # input to discriminator
    input_shape = HIGH_SHAPE    
    input_layer = Input(shape = input_shape)

    # first convolution block
    dis1 = MY_CONV(64, 3, 1)(input_layer)
    dis1 = LeakyReLU(MY_ALPHA)(dis1)

    # second convolution block
    dis2 = MY_CONV(64, 3, 2)(dis1)
    dis2 = LeakyReLU(MY_ALPHA)(dis2)
    dis2 = BatchNormalization(momentum = MY_MOM)(dis2)

    # third convolution block
    dis3 = MY_CONV(128, 3, 1)(dis2)
    dis3 = LeakyReLU(MY_ALPHA)(dis3)
    dis3 = BatchNormalization(momentum = MY_MOM)(dis3)

    # fourth convolution block
    dis4 = MY_CONV(128, 3, 2)(dis3)
    dis4 = LeakyReLU(MY_ALPHA)(dis4)
    dis4 = BatchNormalization(momentum = MY_MOM)(dis4)

    # fifth convolution block
    dis5 = MY_CONV(256, 3, 1)(dis4)
    dis5 = LeakyReLU(MY_ALPHA)(dis5)
    dis5 = BatchNormalization(momentum = MY_MOM)(dis5)

    # sixth convolution block
    dis6 = MY_CONV(256, 3, 2)(dis5)
    dis6 = LeakyReLU(MY_ALPHA)(dis6)
    dis6 = BatchNormalization(momentum = MY_MOM)(dis6)

    # seventh convolution block
    dis7 = MY_CONV(512, 3, 1)(dis6)
    dis7 = LeakyReLU(MY_ALPHA)(dis7)
    dis7 = BatchNormalization(momentum = MY_MOM)(dis7)

    # eight convolution block
    dis8 = MY_CONV(512, 3, 2)(dis7)
    dis8 = LeakyReLU(MY_ALPHA)(dis8)
    dis8 = BatchNormalization(momentum = MY_MOM)(dis8)

    # add a dense layer
    dis9 = Dense(units = 1024)(dis8)
    dis9 = LeakyReLU(MY_ALPHA)(dis9)

    # last dense layer with single output for classification
    output = Dense(units = 1, activation = 'sigmoid')(dis9)

    # final keras model for discriminator
    model = Model(inputs = [input_layer], outputs = [output])

    print('\n== DISCRIMINATOR MODEL SUMMARY ==')
    print('Input shape:', input_shape)
    print('Output shape:', output.shape)
    model.summary()

    return model


# residual block used in generator
def residual_block(x):

    # first convolution block
    res = MY_CONV(64, 3, 1)(x)
    res = Activation(activation = "relu")(res)
    res = BatchNormalization(momentum = MY_MOM)(res)

    # second convolution block
    res = MY_CONV(64, 3, 1)(res)
    res = BatchNormalization(momentum = MY_MOM)(res)

    # add bypass synaptic connections
    res = Add()([res, x])
    return res


# create a generator network using deep CNN
# it has 2 million trainable parameters
def build_generator():
    
    # Input Layer of the generator network
    input_shape = LOW_SHAPE    
    input_layer = Input(shape = input_shape)

    # pre-residual block
    gen1 = Conv2D(64, (9, 9), strides = 1, padding = 'same', 
            activation = 'relu')(input_layer)

    print('\n== GENERATOR MODEL SUMMARY ==')
    print('Input shape:', input_shape)
    print('After pre-residual:', gen1.shape)

    # Add 15 residual blocks
    res = residual_block(gen1)
    for i in range(MY_RESIDUAL):
        res = residual_block(res)
    print('After adding residual:', res.shape)

    # post-residual block
    gen2 = MY_CONV(64, 3, 1)(res)
    gen2 = BatchNormalization(momentum = MY_MOM)(gen2)
    print('After post-residual', gen2.shape)

    # take the sum of the output from pre-residual block (gen1) 
    # and the post-residual block (gen2)
    gen3 = Add()([gen2, gen1])
    print('After adding pre- and post-residual', gen3.shape)

    # add an upsampling block
    gen4 = UpSampling2D(size = 2)(gen3)
    gen4 = MY_CONV(256, 3, 1)(gen4)
    gen4 = Activation('relu')(gen4)
    print('After first upscaling', gen4.shape)

    # add another upsampling block
    gen5 = UpSampling2D(size = 2)(gen4)
    gen5 = MY_CONV(256, 3, 1)(gen5)
    gen5 = Activation('relu')(gen5)
    print('After second upscaling', gen5.shape)

    # output convolution layer
    gen6 = MY_CONV(3, 9, 1)(gen5)
    output = Activation('tanh')(gen6)
    print('Shape of the generator output', output.shape)

    # Keras model of our generator
    model = Model(inputs = [input_layer], outputs = [output])
    model.summary()

    return model


# create GASN network using all preview models including
# VGG19, discriminator, and generator
def build_GAN():

    # build and compile VGG19 network to extract features
    # shape change: (?, 256, 256, 3) -> (?, 64, 64, 256)
    # we do not NOT need compile statement
    vgg = build_vgg()

    # build and compile the discriminator network
    # shape change: (?, 256, 256, 3) -> (?, 16, 16, 1)
    discriminator = build_discriminator()
    discriminator.compile(loss = 'mse', optimizer = MY_OPT, metrics = ['acc'])

    # Build the generator network
    # compilation of generator is the same as that of GAN
    # shape change: (?, 64, 64, 3) -> (?, 256, 256, 3)
    generator = build_generator()

    # GAN has one input
    # for low-resolution images that go to generator
    input_low = Input(shape = LOW_SHAPE)

    # generator accepts the low resolution images
    # and produce fake high resolution images
    fake_high = generator(input_low)

    # 1. discriminator accepts the fake image
    # and compute the probability of fake being close to real
    probs = discriminator(fake_high)

    # 2. VGG19 accepts fake images produced by generator
    # and extract their feature maps
    features = vgg(fake_high)

    # we initially set the discriminator non-trainable
    discriminator.trainable = False

    # Create and compile an adversarial model
    # our GAN has one input and two outputs
    # the input includes:
    #    1. low resolution image (goes to generator)
    # the two output labels to guide backpropagation include: 
    #    1. probs: discriminator output
    #    2. features: VGG output 
    gan_model = Model([input_low], [probs, features])

    # we train generator during GAN training
    # discriminator is not trained
    # loss = 0.001 x entropy (probs, DIS) + 1 x mse (features, GEN)
    gan_model.compile(loss = ['binary_crossentropy', 'mse'], 
            loss_weights = [0.001, 1], optimizer = MY_OPT)


    # add tensorboard to plot loss curves
    # tensorboard --logdir=logs
    # localhost:6006
    tensorboard = TensorBoard(log_dir = LOG_DIR)
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    return vgg, discriminator, generator, gan_model, tensorboard


    ##################
    # MODEL TRAINING #
    ##################


# write train data to tensorboard
def write_log(callback, name, value, epoch):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, epoch)
    callback.writer.flush()


# train discriminator
def train_D():
    # sample a batch of images
    # and normalize the pixel values to [-1, 1]
    # this works well with tanh activation
    real_high, real_low = sample_images(MY_BATCH)
    real_high = real_high / 127.5 - 1.
    real_low = real_low / 127.5 - 1.


    # generate high-resolution images from low-resolution images
    fake_high = generator.predict(real_low)


    # we use 256 values (= 16 x 16) to represent 
    # how realistic fake images are
    # initialize real labels with 1 and fake 0
    real_labels = np.ones(DIS_ANS)
    fake_labels = np.zeros(DIS_ANS)

    # we first train the discriminator using real data
    # train_on_batch accepts (training data, target data)        
    # we get two mse loss values in return:
    # only the first value is relevant
    # use train_on_batch() instead of fit() for GAN
    d_loss_real = discriminator.train_on_batch(real_high, real_labels)


    # we train the discriminator again using fake data and labels
    # we get two mse loss values in return:
    # only the first value is relevant
    # use train_on_batch() instead of fit() for GAN
    d_loss_fake = discriminator.train_on_batch(fake_high, fake_labels)


    # we take the average of the two above and report as the combined loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    print("  Discriminator loss:", d_loss[0])

    return d_loss


# train generator
def train_G():
    # sample a new batch of images
    # and normalize the pixel values to [-1, 1]
    # this works well with tanh activation
    real_high, real_low = sample_images(MY_BATCH)
    real_high = real_high / 127.5 - 1.
    real_low = real_low / 127.5 - 1.


    # Extract feature maps for real high-resolution images
    # VGG shape change: (?, 256, 256, 3) -> (?, 64, 64, 256)
    real_features = vgg.predict(real_high)
    real_labels = np.ones(DIS_ANS)


    # Train the generator network
    # we want generator to fake real images as much as possible
    # discriminator is not trained during this process
    # our GAN needs the following model parameters
    # ([input_low], [probs for DIS out, features for VGG out])
    # use train_on_batch() instead of fit() for GAN
    g_loss = gan_model.train_on_batch([real_low],
            [real_labels, real_features])

    return g_loss


# overall GAN training
# we alternate between discriminator and generator training 
# during generator training, discriminator is not trained
def train_GAN(vgg, discriminator, generator, gan_model, TB):
    print('\n== GAN TRAINING STARTS ==')

    # repeat epochs
    for epoch in range(MY_EPOCH):
        print("Epoch:", epoch)

        d_loss = train_D()
        g_loss = train_G()

        # we print perceptual loss
        # and write the losses to tensorboard
        print("  Generator loss:", g_loss[0])
        write_log(TB, 'g_loss', g_loss[0], epoch)
        write_log(TB, 'd_loss', d_loss[0], epoch)


    # report final loss
    print('\n== FINAL LOSS INFO ==')
    print('Discriminator:', d_loss[0])
    print('Generator:', g_loss[0])


    # Save models
    generator.save_weights("model/chap6-gen.h5")
    discriminator.save_weights("model/chap6-dis.h5")


# save low-resolution, high-resolution (original) and
# fake high-resolution images 
# add_subplot needs (nrows, ncols, index) 
def save_images(low_res_image, original_image, fake_image, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_res_image)
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(fake_image)
    ax.axis("off")
    ax.set_title("Fake")

    plt.savefig(path)


    ####################
    # MODEL EVALUATION #
    ####################


# we pick 5 random images to test our GAN model
# we compare 3 variations for each image:
#    1. low resolution (input)
#    2. high resolution (truth)
#    3. high resolution (fake) 
def evaluate_GAN(generator, discriminator):

    for i in range(5):

        # sample a new batch of images
        # and normalize the pixel values to [-1, 1]
        # this works well with tanh activation
        real_high, real_low = sample_images(MY_BATCH)
        real_high = real_high / 127.5 - 1.
        real_low = real_low / 127.5 - 1.


        # we use generator to turn a low resolution image to high        
        # and save the 3 image files
        fake_image = generator.predict_on_batch(real_low)
        path = "output/chap6-img-{}".format(i)
        real_low = real_low.reshape(LOW_SHAPE)
        real_high = real_high.reshape(HIGH_SHAPE)
        fake_image = fake_image.reshape(HIGH_SHAPE)
        save_images(real_low, real_high, fake_image, path)


# prediction with GAN
def gan_prediction():

    # we just need a trained generator
    generator = build_generator()
    generator.load_weights("model/chap6-gen.h5")

    for i in range(5):
        # sample a new batch of images
        # and normalize the pixel values to [-1, 1]
        # this works well with tanh activation
        real_high, real_low = sample_images(MY_BATCH)
        real_high = real_high / 127.5 - 1.
        real_low = real_low / 127.5 - 1.

        # we use generator to tunr a low resolution image to high        
        # and save the 3 image files
        fake_image = generator.predict_on_batch(real_low)
        path = "output/chap6-img-pred-{}".format(i)
        real_low = real_low.reshape(LOW_SHAPE)
        real_high = real_high.reshape(HIGH_SHAPE)
        fake_image = fake_image.reshape(HIGH_SHAPE)
        save_images(real_low, real_high, fake_image, path)


# if training is not done, we do training and save the models
# otherwise, we use the saved models to do prediction
if not TRAINING_DONE:
    vgg, discriminator, generator, gan_model, tensorboard = build_GAN()
    train_GAN(vgg, discriminator, generator, gan_model, tensorboard)
    evaluate_GAN(generator, discriminator)
else:
    gan_prediction()
    exit()

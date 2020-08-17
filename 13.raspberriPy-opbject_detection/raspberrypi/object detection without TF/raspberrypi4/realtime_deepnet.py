# coding: utf-8
import sys, os
import numpy as np
from deep_convnet import DeepConvNet
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
#
network = DeepConvNet(input_dim=(1, 320, 240),
                       hidden_size=50, output_size=3, optimizer='Adam', learning_rate=0.01, train_flg=True)
network.load_params("deep_convnet_params.pkl")

camera = PiCamera()
camera.resolution = (320, 240)
camera.color_effects = (128, 128)
camera.framerate = 24
camera.start_preview()


for filename in camera.capture_continuous('img{counter:03d}.jpg'):
    print('Captured %s' % filename)
    time.sleep(1) # wait 5 minutes
    
##    camera.resolution = (320, 240)
##    camera.color_effects = (128, 128)
##    camera.framerate = 24
    
    img = np.expand_dims(filename, axis=0)
    img = np.transpose(img, (0, 3, 2, 1))

    start = time.time()
    result = network.predict(img, train_flg=False)
    print(np.argmax(result))
    end = time.time()
    print("time: ", end-start, "second.")



# with PiCamera() as camera:
#     camera.resolution = (320, 240)
#     camera.color_effects = (128, 128)
#     camera.framerate = 24
#     time.sleep(2)
#     img = np.empty((240 ,320, 1,), dtype=np.float32)
#     start = time.time()
#     camera.capture(img, 'rgb')
#     #print(type(img))
#     #print(img.shape)
#     img = np.expand_dims(img, axis=0)
#     img = np.transpose(img, (0, 3, 2, 1))
#     #print(img.shape)
#     #print(img)
#     result = network.predict(img, train_flg=False)
#     print(np.argmax(result))
#     end = time.time()
#     print("time: ", end-start, "second.")
#
#     start = time.time()
#     camera.capture(img, 'rgb')
#     #print(type(img))
#     #print(img.shape)
#     img = np.expand_dims(img, axis=0)
#     img = np.transpose(img, (0, 3, 2, 1))
#     #print(img.shape)
#     #print(img)
#     result = network.predict(img, train_flg=False)
#     print(np.argmax(result))
#     end = time.time()
#     print("time: ", end-start, "second.")
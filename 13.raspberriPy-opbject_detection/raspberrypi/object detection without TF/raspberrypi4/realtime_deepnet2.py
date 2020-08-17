# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from deeplink.mnist import *
from deep_convnet import DeepConvNet

network = DeepConvNet(input_dim=(1, 320, 240),
                      hidden_size=128, output_size=3, optimizer='AdaGrad', learning_rate=0.01, train_flg=False)

network.load_params("deep_convnet_params.pkl")

result = network.predict(img, train_flg=False)
print(np.argmax(result))


# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from deeplink.mnist import *
from deep_convnet import DeepConvNet
import math
import matplotlib.pyplot as plt

"/Users/do-hyungkwon/GoogleDrive/link.dohk@gmail.com/raspberrypi_project/Training_OnDesktop/dataset/image"
data = mnist_data("/Users/do-hyungkwon/GoogleDrive/deeplink/0.Common/1.DeepLearning/data/MNIST_data/.")
(img_train, label_train), (img_validation, label_validation), (img_test, label_test) = data.load_mnist(flatten=False, normalize=True, one_hot_label=True)

num_epochs = 50
train_size = img_train.shape[0]
batch_size = 100
learning_rate = 0.01


network = DeepConvNet(learning_rate = learning_rate)
network.load_params("deep_convnet_params.pkl")

result = network.predict(img)
print(np.argmax(result))


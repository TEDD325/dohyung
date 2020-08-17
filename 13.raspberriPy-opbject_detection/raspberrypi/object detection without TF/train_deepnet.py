# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from deeplink.mnist import *
from deep_convnet import DeepConvNet
import math
import matplotlib.pyplot as plt

# "/Users/do-hyungkwon/GoogleDrive/link.dohk@gmail.com/raspberrypi_project/Training_OnDesktop/dataset/image"
data = mnist_data("/Users/do-hyungkwon/GoogleDrive/deeplink/0.Common/1.DeepLearning/data/MNIST_data/.")
(img_train, label_train), (img_validation, label_validation), (img_test, label_test) = data.load_mnist(flatten=False, normalize=True, one_hot_label=True)

num_epochs = 50
train_size = img_train.shape[0]
batch_size = 100
learning_rate = 0.01


network = DeepConvNet(learning_rate = learning_rate)


train_error_list = []
validation_error_list = []

test_accuracy_list = []
epoch_list = []

num_batch = math.ceil(train_size / batch_size)

for i in range(num_epochs):
    for j in range(num_batch):
        print(j, end=", ")
        sys.stdout.flush()
        x_batch = img_train[j * batch_size: j * batch_size + batch_size]
        t_batch = label_train[j * batch_size: j * batch_size + batch_size]
        network.learning(x_batch, t_batch)

    print()

    epoch_list.append(i)

    train_loss = network.loss(x_batch, t_batch)
    train_error_list.append(train_loss)

    validation_loss = network.loss(img_validation, label_validation)
    validation_error_list.append(validation_loss)

    test_accuracy = network.accuracy(img_test, label_test)
    test_accuracy_list.append(test_accuracy)

    print("Epoch: {0:5d}, Train Error: {1:7.5f}, Validation Error: {2:7.5f} - Test Accuracy: {3:7.5f}".format(
        i,
        train_loss,
        validation_loss,
        test_accuracy
    ))

# Draw Graph about Error Values & Accuracy Values
def draw_error_values_and_accuracy(epoch_list, train_error_list, validation_error_list, test_accuracy_list):
    # Draw Error Values and Accuracy
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(epoch_list[1:], train_error_list[1:], 'r', label='Train')
    plt.plot(epoch_list[1:], validation_error_list[1:], 'g', label='Validation')
    plt.ylabel('Total Error')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.subplot(122)
    plt.plot(epoch_list[1:], test_accuracy_list[1:], 'b', label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(0.0, 1.0, 0.05))
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()

def draw_false_prediction(diff_index_list):
    fig = plt.figure(figsize=(20, 5))
    for i in range(5):
        j = diff_index_list[i]
        print("False Prediction Index: %s, Prediction: %s, Ground Truth: %s" % (j, prediction[j], ground_truth[j]))
        img = np.array(img_test[j])
        img.shape = (28, 28)
        plt.subplot(150 + (i + 1))
        plt.imshow(img, cmap='gray')

draw_error_values_and_accuracy(epoch_list, train_error_list, validation_error_list, test_accuracy_list)

prediction = np.argmax(network.predict(img_test), axis=1)
ground_truth = np.argmax(label_test, axis=1)

print(prediction)
print(ground_truth)

diff_index_list = []
for i in range(len(img_test)):
    if (prediction[i] != ground_truth[i]):
        diff_index_list.append(i)

print("Total Test Image: {0}, Number of False Prediction: {1}".format(len(img_test), len(diff_index_list)))
print("Test Accuracy:", float(len(img_test) - len(diff_index_list)) / float(len(img_test)))
draw_false_prediction(diff_index_list)

network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread
import numpy as np
import os
import tensorflow as tf
import glob
import random
import shutil

IMG_width = 320
IMG_height = 240
IMG_channel = 3
DATASET_PATH = "./dataset_jpg"
kinds = ["train", "validation", "test"]
train_img = np.array([], dtype="float32")
train_label = np.array([], dtype="int8")
validation_img = np.array([], dtype="float32")
validation_label = np.array([], dtype="int8")
test_img = np.array([], dtype="float32")
test_label = np.array([], dtype="int8")
(label_train, filename_train) = (0, 0)
(label_validation, filename_validation) = (0, 0)
(label_test, filename_test) = (0, 0)


def file_info(category_name, dataset_path=DATASET_PATH):
    # 디렉토리 상의 파일경로와 파일의 제일 앞에 매겨진 숫자정보(라벨 정보)를 긁어서 반환
    full_path =  dataset_path + '/' + category_name + '/' + '*.jpg'
    image_filenames = glob.glob(full_path)
    filename = []
    label = []
    for image_filename in image_filenames:
        filename.append(image_filename.split("/")[3])
        label.append(image_filename.split("/")[3].split("-")[0])
    return (label, filename)


# def display_image(image, label):
#     % matplotlib
#     inline

    fig = plt.figure(figsize=(20, 5))
    for i in range(5):
        print(label[i])
        img = image[i]
        img = img.reshape(240, 320, 3)
        img.shape = (240, 320, 3)
        plt.subplot(150 + (i + 1))
        plt.imshow(img)


def data_processing_about_train(idx, kind):
    for i in range(idx):
        global train_img
        global train_label

        (labels, filenames) = file_info(kind)
        random_value = random.randrange(0, len(filenames))
        filename = filenames[random_value]
        label = labels[random_value]
        #         label = _change_one_hot_label(train_label, label)
        #         label = labels[random_value]
        #         image_display(kinds, filename)
        full_path = ( DATASET_PATH + '/' + kind + '/' + filename)
        image = imread(full_path)
        image = image.reshape(IMG_width * IMG_height * IMG_channel) / 255.0
        train_img = np.append(train_img, image)
        label = _change_one_hot_label(label)
        train_label = np.append(train_label, label)
        # file move
        # 학습한 데이터는 이동시킴
        print("---------")
        print(len(filenames))
        print(filename)
        src =  DATASET_PATH + '/' + kind + '/'
        dir =  DATASET_PATH + '/' + 'tmp' + '/' + kind + '/'
        shutil.move(src + filename, dir + filename)


def data_processing_about_validation(idx, kind):
    for i in range(idx):
        global validation_img
        global validation_label

        (labels, filenames) = file_info(kind)
        random_value = random.randrange(0, len(filenames))
        filename = filenames[random_value]
        label = labels[random_value]
        #         label = _change_one_hot_label(train_label, label)
        #         label = labels[random_value]
        #         image_display(kinds, filename)
        full_path = ( DATASET_PATH + '/' + kind + '/' + filename)
        image = imread(full_path)
        image = image.reshape(IMG_width * IMG_height * IMG_channel) / 255.0
        validation_img = np.append(validation_img, image)
        label = _change_one_hot_label(label)
        validation_label = np.append(validation_label, label)
        # file move
        print("---------")
        print(len(filenames))
        print(filename)
        src =  DATASET_PATH + '/' + kind + '/'
        dir =  DATASET_PATH + '/' + 'tmp' + '/' + kind + '/'
        shutil.move(src + filename, dir + filename)


def data_processing_about_test(idx, kind):
    for i in range(idx):
        global test_img
        global test_label

        (labels, filenames) = file_info(kind)
        random_value = random.randrange(0, len(filenames))
        filename = filenames[random_value]
        label = labels[random_value]
        #         label = _change_one_hot_label(train_label, label)
        #         label = labels[random_value]
        #         image_display(kinds, filename)
        full_path = ( DATASET_PATH + '/' + kind + '/' + filename)
        image = imread(full_path)
        image = image.reshape(IMG_width * IMG_height * IMG_channel) / 255.0
        test_img = np.append(test_img, image)
        label = _change_one_hot_label(label)
        test_label = np.append(test_label, label)
        # file move
        print("---------")
        print(len(filenames))
        print(filename)
        src =  DATASET_PATH + '/' + kind + '/'
        dir =  DATASET_PATH + '/' + 'tmp' + '/' + kind + '/'
        shutil.move(src + filename, dir + filename)


def _change_one_hot_label(target_label):
    target_label = int(target_label)
    T = np.zeros((1, 3))
    T[0][target_label] = 1

    return T


def store_dataset(dataset_path=DATASET_PATH):
    global train_img
    global train_label
    global validation_img
    global validation_label
    global test_img
    global test_label

    (_, idx_train) = file_info("train")
    (_, idx_validation) = file_info("validation")
    (_, idx_test) = file_info("test")
    if len(idx_train) != 0 or len(idx_validation) != 0 or len(idx_test) != 0:
        data_processing_about_train(len(idx_train), "train")
        data_processing_about_validation(len(idx_validation), "validation")
        data_processing_about_test(len(idx_test), "test")

        # file로 쓰기
        train_img.tofile( dataset_path + '/' + 'train_img_dataset.txt')
        train_label.tofile( dataset_path + '/' + 'train_label.txt')
        validation_img.tofile( dataset_path + '/' + 'validation_img_dataset.txt')
        validation_label.tofile( dataset_path + '/' + 'validation_label.txt')
        test_img.tofile(dataset_path + '/' + 'test_img_dataset.txt')
        test_label.tofile(dataset_path + '/' + 'test_label.txt')

    else:
        print("[!] Already Finished Generateing Dataset. Please check directory.")





def load_dataset(dataset_path):
    (img_train, label_train) = (0,0)
    (img_validation, label_validation) = (0,0)
    (img_test, label_test) = (0,0)
    filenames = glob.glob(dataset_path + "/" + "*.txt")
    filename = []
    for filename in filenames:
        print(filename, )
        if filename.split("/")[6][-4:] == ".txt": # CAUTION : filename.split("/")[directory level][-4:]
            img_train = np.fromfile(open(dataset_path + "/" + "train_img_dataset.txt", 'rb')).reshape(451,IMG_width * IMG_height * IMG_channel)
            label_train = np.fromfile(open(dataset_path + "/" + "train_label.txt", 'rb')).reshape(451, 3)
            img_validation = np.fromfile(open(dataset_path + "/" + "validation_img_dataset.txt", 'rb')).reshape(65, IMG_width * IMG_height * IMG_channel)
            label_validation = np.fromfile(open(dataset_path + "/" + "validation_label.txt", 'rb')).reshape(65,3)
            img_test = np.fromfile(open(dataset_path + "/" + "test_img_dataset.txt", 'rb')).reshape(130,IMG_width * IMG_height * IMG_channel)
            label_test = np.fromfile(open(dataset_path + "/" + "test_label.txt", 'rb')).reshape(130, 3)
    return ((img_train, label_train), (img_validation, label_validation), (img_test, label_test))





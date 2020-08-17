# coding: utf-8
import numpy as np
import glob
import random
import shutil
import matplotlib.pyplot as plt
from scipy.misc import imread

class CustomImageData:
    def __init__(self, dataset_path, image_width=320, image_height=240, image_channel=1):
        self.dataset_path = dataset_path
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        self.kinds = ["train", "validation", "test"]
        self.train_img = np.array([], dtype="float32")
        self.train_label = np.array([], dtype="int8")
        self.validation_img = np.array([], dtype="float32")
        self.validation_label = np.array([], dtype="int8")
        self.test_img = np.array([], dtype="float32")
        self.test_label = np.array([], dtype="int8")

    def file_info(self, category_name, dataset_path):
        # 디렉토리 상의 파일경로와 파일의 제일 앞에 매겨진 숫자정보(라벨 정보)를 긁어서 반환
        full_path = self.dataset_path + '/' + category_name + '/' + '*.jpg'
        image_filenames = glob.glob(full_path)
        filename = []
        label = []
        for image_filename in image_filenames:
            filename.append(image_filename.split("/")[3])
            label.append(image_filename.split("/")[3].split("-")[0])
        return (label, filename)

    def display_image(self, image, label):
        fig = plt.figure(figsize=(20, 5))
        for i in range(5):
            print(label[i])
            img = image[i]
            img = img.reshape(self.image_width, self.image_height)
            img.shape = (self.image_width, self.image_height)
            plt.subplot(150 + (i + 1))
            plt.imshow(img)

    def data_processing_about_train(self, dataset_path, idx, kind):
        for i in range(idx):
            (labels, filenames) = self.file_info(kind)
            random_value = random.randrange(0, len(filenames))
            filename = filenames[random_value]
            label = labels[random_value]
            #         label = self._change_one_hot_label(self.train_label, label)
            #         label = labels[random_value]
            #         image_display(self.kinds, filename)
            full_path = (self.dataset_path + '/' + kind + '/' + filename)
            image = imread(full_path)
            image = image.reshape(self.image_width * self.image_height * self.image_channel) / 255.0
            self.train_img = np.append(self.train_img, image)
            label = self._change_one_hot_label(label)
            self.train_label = np.append(self.train_label, label)
            # file move
            # 학습한 데이터는 이동시킴
            print("---------")
            print(len(filenames))
            print(filename)
            src = self.dataset_path + '/' + kind + '/'
            dir = self.dataset_path + '/' + 'tmp' + '/' + kind + '/'
            shutil.move(src + filename, dir + filename)


    def data_processing_about_validation(self, dataset_path, idx, kind):
        for i in range(idx):
            (labels, filenames) = self.file_info(kind)
            random_value = random.randrange(0, len(filenames))
            filename = filenames[random_value]
            label = labels[random_value]
            #         label = self._change_one_hot_label(self.train_label, label)
            #         label = labels[random_value]
            #         image_display(self.kinds, filename)
            full_path = (self.dataset_path + '/' + kind + '/' + filename)
            image = imread(full_path)
            image = image.reshape(self.image_width * self.image_height * self.image_channel) / 255.0
            self.validation_img = np.append(self.validation_img, image)
            label = self._change_one_hot_label(label)
            self.validation_label = np.append(self.validation_label, label)
            # file move
            print("---------")
            print(len(filenames))
            print(filename)
            src = self.dataset_path + '/' + kind + '/'
            dir = self.dataset_path + '/' + 'tmp' + '/' + kind + '/'
            shutil.move(src + filename, dir + filename)

    def data_processing_about_test(self, idx, kind):
        for i in range(idx):
            (labels, filenames) = self.file_info(kind)
            random_value = random.randrange(0, len(filenames))
            filename = filenames[random_value]
            label = labels[random_value]
            #         label = self._change_one_hot_label(self.train_label, label)
            #         label = labels[random_value]
            #         image_display(self.kinds, filename)
            full_path = (self.dataset_path + '/' + kind + '/' + filename)
            image = imread(full_path)
            image = image.reshape(self.image_width * self.image_height * self.image_channel) / 255.0
            self.test_img = np.append(self.test_img, image)
            label = self._change_one_hot_label(label)
            self.test_label = np.append(self.test_label, label)
            # file move
            print("---------")
            print(len(filenames))
            print(filename)
            src = self.dataset_path + '/' + kind + '/'
            dir = self.dataset_path + '/' + 'tmp' + '/' + kind + '/'
            shutil.move(src + filename, dir + filename)

    def _change_one_hot_label(self, target_label):
        target_label = int(target_label)
        T = np.zeros((1, 3))
        T[0][target_label] = 1

        return T

    def store_dataset(self, dataset_path):
        (_, idx_train) = self.file_info("train")
        (_, idx_validation) = self.file_info("validation")
        (_, idx_test) = self.file_info("test")
        if len(idx_train) != 0 or len(idx_validation) != 0 or len(idx_test) != 0:
            self.data_processing_about_train(dataset_path, len(idx_train), "train")
            self.data_processing_about_validation(dataset_path, len(idx_validation), "validation")
            self.data_processing_about_test(len(idx_test), "test")

            # file로 쓰기
            self.train_img.tofile(self.dataset_path + '/' + 'train_img_dataset')
            self.train_label.tofile(self.dataset_path + '/' + 'train_label')
            self.validation_img.tofile(self.dataset_path + '/' + 'validation_img_dataset')
            self.validation_label.tofile(self.dataset_path + '/' + 'validation_label')
            self.test_img.tofile(self.dataset_path + '/' + 'test_img_dataset')
            self.test_label.tofile(self.dataset_path + '/' + 'test_label')
        else:
            print("[!] Already Finished Generateing Dataset. Please check directory.")

    def load_dataset(self, flatten=True):
        (img_train, label_train) = (0,0)
        (img_validation, label_validation) = (0,0)
        (img_test, label_test) = (0,0)
        filenames = glob.glob(self.dataset_path + "/*")
        filename = []
        for filename in filenames:
            print(filename, )
            img_train = np.fromfile(open(self.dataset_path + "/" + "train_img_dataset", 'rb')).reshape(451,self.image_width * self.image_height * self.image_channel)
            label_train = np.fromfile(open(self.dataset_path + "/" + "train_label", 'rb')).reshape(451, 3)
            img_validation = np.fromfile(open(self.dataset_path + "/" + "validation_img_dataset", 'rb')).reshape(65, self.image_width * self.image_height * self.image_channel)
            label_validation = np.fromfile(open(self.dataset_path + "/" + "validation_label", 'rb')).reshape(65,3)
            img_test = np.fromfile(open(self.dataset_path + "/" + "test_img_dataset", 'rb')).reshape(130,self.image_width * self.image_height * self.image_channel)
            label_test = np.fromfile(open(self.dataset_path + "/" + "test_label", 'rb')).reshape(130, 3)

        if not flatten:
            img_train      = img_train.reshape(-1, 1, self.image_width, self.image_height)
            img_validation = img_validation.reshape(-1, 1, self.image_width, self.image_height)
            img_test       = img_test.reshape(-1, 1, self.image_width, self.image_height)

        return ((img_train, label_train), (img_validation, label_validation), (img_test, label_test))
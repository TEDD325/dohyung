# coding: utf-8
import sys, os
import pickle
import numpy as np
from deeplink.layers import *
from deeplink.optimizers import *

optimizers = {
    "SGD": SGD,
    "Momentum": Momentum,
    "Nesterov": Nesterov,
    "AdaGrad": AdaGrad,
    "RMSprop": RMSprop,
    "Adam": Adam
}

class DeepConvNet:
    """
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 320, 240),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':4, 'pad':2, 'stride':1},
                 hidden_size=50, output_size=3, optimizer = 'Adam', learning_rate=0.01, train_flg=False):

        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*41*31, hidden_size])

        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}

        pre_channel_num = input_dim[0]
        layer_input_width = input_dim[1]
        layer_input_height = input_dim[2]

        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(ReLU())

        if train_flg:
            layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                conv_param_1['filter_size'],
                                                                                conv_param_1['pad'], conv_param_1['stride'])
            print('conv-1', layer_input_width, layer_input_height)

        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(ReLU())

        if train_flg:
            layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                conv_param_2['filter_size'],
                                                                                conv_param_2['pad'], conv_param_2['stride'])
            print('conv-2', layer_input_width, layer_input_height)

        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        if train_flg:
            layer_input_width, layer_input_height = self.pool_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                2, 2, 2)
            print('pool-1', layer_input_width, layer_input_height)

        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(ReLU())

        if train_flg:
            layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                conv_param_3['filter_size'],
                                                                                conv_param_3['pad'], conv_param_3['stride'])
            print('conv-3', layer_input_width, layer_input_height)

        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(ReLU())

        if train_flg:
            layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                conv_param_4['filter_size'],
                                                                                conv_param_4['pad'], conv_param_4['stride'])
            print('conv-4', layer_input_width, layer_input_height)

        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        if train_flg:
            layer_input_width, layer_input_height = self.pool_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                2, 2, 2)
            print('pool-2', layer_input_width, layer_input_height)

        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(ReLU())

        if train_flg:
            layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                conv_param_5['filter_size'],
                                                                                conv_param_5['pad'], conv_param_5['stride'])
            print('conv-5', layer_input_width, layer_input_height)

        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(ReLU())

        if train_flg:
            layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                conv_param_6['filter_size'],
                                                                                conv_param_6['pad'], conv_param_6['stride'])
            print('conv-6', layer_input_width, layer_input_height)

        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        if train_flg:
            layer_input_width, layer_input_height = self.pool_layer_output_size(layer_input_width,
                                                                                layer_input_height,
                                                                                2, 2, 2)
            print('pool-3', layer_input_width, layer_input_height)

        layer_input_width = int(layer_input_width)
        layer_input_height = int(layer_input_height)

        print(64 * layer_input_width * layer_input_height)

        self.last_conv_pool_input_width = layer_input_width
        self.last_conv_pool_input_height = layer_input_height

        self.params['W7'] = weight_init_scales[6] * np.random.randn(64 * layer_input_width * layer_input_height, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithCrossEntropyLoss()

        # Optimizer Initialization
        self.optimizer = optimizers[optimizer](lr=learning_rate)

    def conv_layer_output_size(self, layer_input_width, layer_input_height, filter_size, pad, stride):
        return (layer_input_width - filter_size + 2 * pad) / stride + 1, (layer_input_height - filter_size + 2 * pad) / stride + 1

    def pool_layer_output_size(self, layer_input_width, layer_input_height, filter_width, filter_height, stride):
        return (layer_input_width - filter_width) / stride + 1, (layer_input_height - filter_height) / stride + 1

    def predict(self, x, train_flg=False):
        isFirstAffine = False
        for layer in self.layers:
            if isinstance(layer, Affine) and not isFirstAffine:
                isFirstAffine = True
                x = x.reshape(-1, 64 * self.last_conv_pool_input_width * self.last_conv_pool_input_height)

            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=10):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def backpropagation_gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()

        isFirstPooling = False
        for layer in tmp_layers:
            if isinstance(layer, Pooling) and not isFirstPooling:
                isFirstPooling = True
                dout = dout.reshape(-1, 64, self.last_conv_pool_input_width, self.last_conv_pool_input_height)
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def learning(self, x_batch, t_batch):
        grads = self.backpropagation_gradient(x_batch, t_batch)
        self.optimizer.update(self.params, grads)

    def save_params(self, file_name="params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]



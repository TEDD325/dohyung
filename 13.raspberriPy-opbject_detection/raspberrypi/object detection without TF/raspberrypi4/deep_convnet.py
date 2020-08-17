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
        conv - relu - poll - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 320, 240),
                 conv_param_1 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 pool_param_1 = {'size': 2, 'stride': 2},
                 conv_param_2 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 pool_param_2 = {'size': 2, 'stride': 2},
                 hidden_size=128, output_size=3, optimizer='AdaGrad', learning_rate=0.01, train_flg=False):

        channel_num = input_dim[0]
        layer_input_width = input_dim[1]
        layer_input_height = input_dim[2]

        self.last_conv_pool_input_width, self.last_conv_pool_input_height = self.get_conv_pool_last_output_size(input_dim[1], input_dim[2], conv_param_1, pool_param_1, conv_param_2, pool_param_2, train_flg)

        pre_node_nums = np.array([
            channel_num * conv_param_1['filter_size'] * conv_param_1['filter_size'],
            conv_param_1['filter_num'] * conv_param_2['filter_size'] * conv_param_2['filter_size'],
            conv_param_2['filter_num'] * self.last_conv_pool_input_width * self.last_conv_pool_input_height,
            hidden_size])

        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}

        for idx, conv_param in enumerate([conv_param_1, conv_param_2]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            channel_num = conv_param['filter_num']

        self.params['W3'] = weight_init_scales[2] * np.random.randn(16 * self.last_conv_pool_input_width * self.last_conv_pool_input_height, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_scales[3] * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_h=pool_param_1['size'], pool_w=pool_param_1['size'], stride=pool_param_1['stride']))
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_h=pool_param_2['size'], pool_w=pool_param_2['size'], stride=pool_param_2['stride']))
        self.layers.append(Affine(self.params['W3'], self.params['b3']))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W4'], self.params['b4']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithCrossEntropyLoss()

        # Optimizer Initialization
        self.optimizer = optimizers[optimizer](lr=learning_rate)

    def get_conv_pool_last_output_size(self, layer_input_width, layer_input_height, conv_param_1, pool_param_1, conv_param_2, pool_param_2, train_flg):
        layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                            layer_input_height,
                                                                            conv_param_1['filter_size'],
                                                                            conv_param_1['pad'], conv_param_1['stride'])

        if train_flg: print("Shape of Conv-1 Output: ({:d}, {:d}, {:d})".format(conv_param_1['filter_num'], layer_input_width, layer_input_height))

        layer_input_width, layer_input_height = self.pool_layer_output_size(layer_input_width,
                                                                            layer_input_height,
                                                                            pool_param_1['size'],
                                                                            pool_param_1['size'],
                                                                            pool_param_1['stride'])
        if train_flg: print("Shape of Pool-1 Output: ({:d}, {:d}, {:d})".format(conv_param_1['filter_num'], layer_input_width, layer_input_height))

        layer_input_width, layer_input_height = self.conv_layer_output_size(layer_input_width,
                                                                            layer_input_height,
                                                                            conv_param_2['filter_size'],
                                                                            conv_param_2['pad'], conv_param_2['stride'])
        if train_flg: print("Shape of Conv-2 Output: ({:d}, {:d}, {:d})".format(conv_param_2['filter_num'], layer_input_width, layer_input_height))

        layer_input_width, layer_input_height = self.pool_layer_output_size(layer_input_width,
                                                                            layer_input_height,
                                                                            pool_param_2['size'],
                                                                            pool_param_2['size'],
                                                                            pool_param_2['stride'])
        if train_flg: print("Shape of Pool-2 Output: ({:d}, {:d}, {:d})".format(conv_param_2['filter_num'], layer_input_width, layer_input_height))
        return int(layer_input_width), int(layer_input_height)

    def conv_layer_output_size(self, layer_input_width, layer_input_height, filter_size, pad, stride):
        size_w, size_h = (layer_input_width - filter_size + 2 * pad) / stride + 1, (layer_input_height - filter_size + 2 * pad) / stride + 1
        if size_w != int(size_w):
            raise ValueError("Conv_Layer output size (size_w) is not integer")
        if size_h != int(size_h):
            raise ValueError("Conv_Layer output size (size_h) is not integer")
        return int(size_w), int(size_h)

    def pool_layer_output_size(self, layer_input_width, layer_input_height, filter_width, filter_height, stride):
        size_w, size_h = (layer_input_width - filter_width) / stride + 1, (layer_input_height - filter_height) / stride + 1
        if size_w != int(size_w):
            raise ValueError("Pool_Layer output size (size_w) is not integer")
        if size_h != int(size_h):
            raise ValueError("Pool_Layer output size (size_h) is not integer")
        return int(size_w), int(size_h)

    def predict(self, x, train_flg=False):
        isFirstAffine = False
        for layer in self.layers:
            if isinstance(layer, Affine) and not isFirstAffine:
                isFirstAffine = True
                x = x.reshape(-1, 16 * self.last_conv_pool_input_width * self.last_conv_pool_input_height)

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
                dout = dout.reshape(-1, 16, self.last_conv_pool_input_width, self.last_conv_pool_input_height)
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 3, 6, 9)):
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

        for i, layer_idx in enumerate((0, 3, 6, 9)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]



import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(64,64), 
                 activation=F.relu, 
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size) 
                                # 첫 번째 히든 레이어에 대한 iter: (4, 64)
                                # 두 번째 히든 레이어에 대한 iter: (64, 64)
            in_size = next_size
            self.hidden_layers.append(fc) # 그냥 파이썬의 list같은 듯.

        # Set output layers
        self.output_layer = nn.Linear(in_size, self.output_size)
                                # 두 버째 히든 레이어에 이은 아웃풋 레이어에 대하여, (64, 2)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x

        pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
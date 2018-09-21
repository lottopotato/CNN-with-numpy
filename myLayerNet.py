import os
import numpy as np

from Layers import *
from collections import OrderedDict
from Function.functions import softmax

class myLayerNet:

     def __init__(self, input_size, n_class, drop_ratio = 0.5):
          self.drop_ratio = drop_ratio
          
          self.para = {}
          # output, input, h, w
          # convolution 4
          self.para['W1'] = np.random.randn(16, 3 ,3, 3) / np.sqrt(3/2)
          self.para['b1'] = np.zeros(16)
               
          self.para['W2'] = np.random.randn(32, 16, 3, 3 ) / np.sqrt(16/2)
          self.para['b2'] = np.zeros(32)

          self.para['W3'] = np.random.randn(64, 32, 3, 3 ) / np.sqrt(32/2)
          self.para['b3'] = np.zeros(64)

          self.para['W4'] = np.random.randn(128, 64, 3, 3 ) / np.sqrt(64/2)
          self.para['b4'] = np.zeros(128)
               
          # output, input
          # fc 2
          self.para['W5'] = np.random.randn(1024, 128 * 2 * 2).transpose(1, 0) / np.sqrt((128 * 2 * 2)/2)
          self.para['b5'] = np.zeros(1024)
               
          self.para['W6'] = np.random.randn(n_class, 1024 * 1 * 1).transpose(1, 0) / np.sqrt((1024 * 1 * 1)/2)
          self.para['b6'] = np.zeros(n_class)
          #============================================#
          
          self.layers = OrderedDict()
          self.layers['C1'] = Convolution(self.para['W1'], self.para['b1'], pad=1)
          self.layers['R1'] = Relu()
          self.layers['P1'] = Pooling(2, stride=2)

          self.layers['C2'] = Convolution(self.para['W2'], self.para['b2'], pad=1)
          self.layers['R2'] = Relu()
          self.layers['P2'] = Pooling(2, stride=2)

          self.layers['C3'] = Convolution(self.para['W3'], self.para['b3'], pad=1)
          self.layers['R3'] = Relu()
          self.layers['D3'] = Drop_out(self.drop_ratio)
          self.layers['P3'] = Pooling(2, stride=2)
          
          self.layers['C4'] = Convolution(self.para['W4'], self.para['b4'], pad=1)
          self.layers['R4'] = Relu()
          self.layers['D4'] = Drop_out(self.drop_ratio)
          self.layers['P4'] = Pooling(2, stride=2)

          self.layers['fc5'] = Affine(self.para['W5'], self.para['b5'])
          self.layers['R5'] = Relu()
          self.layers['D5'] = Drop_out(self.drop_ratio)
          
          self.layers['fc6'] = Affine(self.para['W6'], self.para['b6'])

          self.lastLayer = SoftmaxWithLoss()

          self.con_len = 4
          self.fc_len = 2
          self.layer_len = self.con_len + self.fc_len

     def predict(self, x):
          for key, layer in self.layers.items():
               x = layer.forward(x)
               #print(x)
               #print(str(key))

          return x

     def loss(self, x, t):
          for i in range(3,6):
               self.layers['D' + str(i)].ratio = 0.5
          y = self.predict(x)
          #print(y)
          return self.lastLayer.forward(y, t)

     def accuracy(self, x, t):
          for i in range(3,6):
               self.layers['D' + str(i)].ratio = 0
          y = self.predict(x)
          y = np.argmax(y, axis=1)
          if t.ndim != 1 :
               t = np.argmax(t, axis=1)

          accuracy = np.sum( y == t ) / float(x.shape[0])

          return accuracy

     def checking(self, x):
          for i in range(3,6):
               self.layers['D' + str(i)].ratio = 0
          y = self.predict(x)

          return softmax(y)

     def gradient(self, x, t):
          self.loss(x, t)

          dout = 1
          dout = self.lastLayer.backward(dout)
          

          b_layers = reversed(list(self.layers.items()))
          for key, layer in b_layers:
               dout = layer.backward(dout)
               #print(str(np.mean(dout)))
               #print(str(key))

          grads = {}
          for i in range(1, self.con_len+1):
               grads['W' + str(i)] = self.layers["C" + str(i)].dW
               grads['b' + str(i)] = self.layers["C" + str(i)].db
          for i in range(self.con_len+1, self.layer_len+1):
               grads['W' + str(i)] = self.layers["fc" + str(i)].dW
               grads['b' + str(i)] = self.layers["fc" + str(i)].db

          return grads


               
          


























                  

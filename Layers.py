import numpy as np
from Function.functions import *
from Function.util import *

class Relu:
     def __init__(self):
          self.mask = None

     def forward(self, x):
          self.mask = ( x <= 0 )
          out = x.copy()
          out[self.mask] = 0
          # if x<=0 then 0, else x
          #print(out)
          return out

     def backward(self, dout):
          dout[self.mask] = 0
          dx = dout
          
          return dx

class leaky_Relu:
     def __init__(self):
          self.mask = None

     def forward(self, x):
          self.mask = ( x <= 0 )
          out = x.copy()
          out2 = np.ones(out.shape)
          out2[self.mask] = 0.01
          out *= out2
          return out

     def backward(self, dout):
          dout[self.mask] = 0.01
          dx = dout
          return dx

class Sigmoid:
     def __init__(self):
          self.out = None

     def forward(self, x):
          out = 1 / (1 + np.exp(-x))
          self.out = out
          return out

     def backward(self, dout):
          dx = dout * (1.0 - self.out) * self.out
          return dx

class Affine:
     def __init__(self, W, b):
          self.W = W
          self.b = b
          self.dW = None
          self.db = None
          self.x = None
          self.original_x_shape = None

     def forward(self, x):
          self.original_x_shape = x.shape
          x = x.reshape(x.shape[0], -1)
          self.x = x
          out = np.dot(self.x, self.W) + self.b

          return out

     def backward(self, dout):
          dx = np.dot(dout, self.W.T)
          self.dW = np.dot(self.x.T, dout)
          self.db = np.sum(dout, axis=0)

          dx = dx.reshape(*self.original_x_shape)
          return dx

class SoftmaxWithLoss:
     def __init__(self):
          self.loss = None
          self.y = None
          self.t = None

     def forward(self, x, t):
          self.t = t
          self.y = softmax(x)
          self.loss = cross_entropy_error(self.y, self.t)
          return self.loss

     def backward(self, dout=1):
          batch_size = self.t.shape[0]
          dx = (self.y - self.t) / batch_size
          return dx

class Pooling:
     def __init__(self, pool, stride=1, pad=0):
          self.pool = pool
          self.stride = stride
          self.pad = pad

          self.x = None
          self.arg_max = None

     def forward(self, x):
          N, C, H, W = x.shape
          out_h = int(( H + 2*self.pad - self.pool) / self.stride + 1)
          out_w = int(( W + 2*self.pad - self.pool) / self.stride + 1)

          col = im2col(x, self.pool, self.pool, self.stride, self.pad)
          col = col.reshape(-1, int(self.pool * self.pool))

          arg_max = np.argmax(col, axis=1)
          out = np.max(col, axis=1) # maxpooling

          out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # N, C, out_h, out_w

          self.x = x
          self.arg_max = arg_max

          #print(out.shape)
          return out

     def backward(self, dout):
          dout = dout.transpose(0, 2, 3, 1)

          pool_size = self.pool * self.pool
          dmax = np.zeros((dout.size, pool_size))
          dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
          dmax = dmax.reshape(dout.shape + (pool_size, ))

          dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
          dx = col2im(dcol, self.x.shape, self.pool, self.pool, self.stride, self.pad)

          #print(dx.shape)
          return dx

class Convolution:
     def __init__ (self, W, b, stride = 1, pad = 0):
          self.W = W
          self.b = b
          self.stride = stride
          self.pad= pad

          self.x = None
          self.col = None
          self.col_W = None

          self.dW = None
          self.db = None

     def forward(self, x):
          FN, C, FH, FW = self.W.shape
          N, C, H, W = x.shape
               
          out_h = int((H + 2*self.pad - FH) / self.stride + 1)
          out_w = int((W + 2*self.pad - FW) / self.stride + 1)

          col = im2col(x, FH, FW, self.stride, self.pad)
          col_W = self.W.reshape(FN, -1).T
          
          out = np.dot(col, col_W) + self.b

          out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
          
          self.x = x
          self.col = col
          self.col_W = col_W
          
          #print("sample w: " + str(np.mean(self.W)))
          return out

     def backward(self, dout):
          FN, C, FH, FW = self.W.shape
          dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

          self.db = np.sum(dout, axis=0)
          self.dW = np.dot(self.col.T, dout)
          self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
          
          dcol = np.dot(dout, self.col_W.T)
          
          dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
          
          #print("sample dW : " + str(np.mean(self.dW)))
          #print(dx)
          return dx

class Drop_out:
     def __init__(self, ratio):
          self.ratio = ratio
          self.mask = None
          
     def forward(self, x):
          self.mask = np.random.rand(*x.shape) > self.ratio
          out = x * self.mask
          
          return out

     def backward(self, dout):
          dx = dout * self.mask
          
          return dx

     

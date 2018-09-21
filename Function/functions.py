import numpy as np

def sigmoid(x):
     return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
     return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
     return np.maximum(0,x)

def d_relu(x):
     grad = np.zeros(x)
     grad[x >= 0] = 1
     return grad

def softmax(x):
     if x.ndim == 2:
          exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
          return exp_x / np.sum(exp_x, axis=1, keepdims=True)
           
     exp_x = np.exp(x - np.max(x))
     return exp_x / np.sum(np.exp(x))
     

def cross_entropy_error(y, t):
     if y.ndim == 1:
          t = t.reshape(1, t.size)
          y = y.reshape(1, y.size)

     if t.size == y.size:
          t = t.argmax(axis=1)

     batch_size = y.shape[0]
     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(x, t):
     y = softmax(x)
     return cross_entropy_error(y, t)

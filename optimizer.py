import numpy as np

class Adam:
     #https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
     #https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py
     def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-08, lr_rate=0.001):
          self.beta1 = beta1
          self.beta2 = beta2
          
          self.w_m = None
          self.w_v = None
          self.b_m = None
          self.b_v = None
          
          self.t = 0
          self.epsilon = epsilon
          self.lr_rate = lr_rate

     def update(self, grads, network):
          if self.w_m == None:
               # must remember prevoius training info
               self.w_m = {}
               self.w_v = {}
               self.b_m = {}
               self.b_v = {}

               for i in range(1, network.layer_len+1):
                    self.w_m[i] = np.zeros((grads['W' + str(i)].shape))
                    self.w_v[i] = np.zeros((grads['W' + str(i)].shape))
                    self.b_m[i] = np.zeros((grads['b' + str(i)].shape))
                    self.b_v[i] = np.zeros((grads['b' + str(i)].shape))
               
          self.t += 1       
          lr_t = self.lr_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
          
          for i in range(1, network.layer_len+1):
               self.w_m[i] = self.beta1 * self.w_m[i] + (1 - self.beta1) * grads['W' + str(i)]
               self.b_m[i] = self.beta1 * self.b_m[i] + (1 - self.beta1) * grads['b' + str(i)]
               self.w_v[i] = self.beta2 * self.w_v[i] + (1 - self.beta2) * grads['W' + str(i)] * grads['W' + str(i)]
               self.b_v[i] = self.beta2 * self.b_v[i] + (1 - self.beta2) * grads['b' + str(i)] * grads['b' + str(i)]
               
               if (i<network.con_len+1):
                    network.layers["C" + str(i)].W -= lr_t * self.w_m[i] / (np.sqrt(self.w_v[i]) + self.epsilon)
                    network.layers["C" + str(i)].b -= lr_t * self.b_m[i] / (np.sqrt(self.b_v[i]) + self.epsilon)
               else:
                    network.layers["fc" + str(i)].W -= lr_t * self.w_m[i] / (np.sqrt(self.w_v[i]) + self.epsilon)
                    network.layers["fc" + str(i)].b -= lr_t * self.b_m[i] / (np.sqrt(self.b_v[i]) + self.epsilon)     
                    
class Momentum:
     def __init__(self, lr_rate = 0.01, mt = 0.9):
          self.lr_rate = lr_rate
          self.mt = mt
          self.w_v = None
          self.b_v = None

     def update(self, grads, network):
          if self.w_v == None:
               self.w_v = {}
               self.b_v = {}
               for i in range(1, network.layer_len+1):
                    self.w_v[i] = np.zeros((grads['W' + str(i)].shape))
                    self.b_v[i] = np.zeros((grads['b' + str(i)].shape))
          for i in range(1, network.layer_len+1):
               self.w_v[i] = self.mt * self.w_v[i] + self.lr_rate * grads['W' + str(i)]
               self.b_v[i] = self.mt * self.b_v[i] + self.lr_rate * grads['b' + str(i)]
               if (i<network.con_len+1):
                    network.layers["C" + str(i)].W -= self.w_v[i]
                    network.layers["C" + str(i)].b -= self.b_v[i]
               else:
                    network.layers["fc" + str(i)].W -= self.w_v[i]
                    network.layers["fc" + str(i)].b -= self.b_v[i]

class SGD:
     def __init__(self, lr_rate = 0.01):
          self.lr_rate = lr_rate
     def update(self, grads, network):
          for i in range(1, network.con_len+1):
               network.layers["C" + str(i)].W -= self.lr_rate * grads['W' + str(i)]
               network.layers["C" + str(i)].b -= self.lr_rate * grads['b' + str(i)]
          for i in range(network.con_len+1, network.layer_len+1):
               network.layers["fc" + str(i)].W -= self.lr_rate * grads['W' + str(i)]
               network.layers["fc" + str(i)].b -= self.lr_rate * grads['b' + str(i)]


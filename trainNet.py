import numpy as np
import time

from Function.util import graph
from Function.save_load_parameter import *
from optimizer import *

class train:
     def __init__(self, data, network, epoch, batch_size, optimizer_name = "Adam", lr_rate = 0.001):          
          self.lr_rate = lr_rate
          self.data = data
          self.network = network
          self.epoch = epoch
          self.batch_size = batch_size
          self.optimizer_name = optimizer_name

          self.train_loss_list = []
          self.train_acc_list = []
          self.valid_acc_list = []

          self.label_list = []

          # Adam optimizer
          if self.optimizer_name == "Adam":    
               self.optimizer = Adam(self.lr_rate)
          # Momentom optimizer
          elif self.optimizer_name == "Momentum":               
               self.optimizer = Momentum(self.lr_rate)
          # SGD
          elif self.optimizer_name == "SGD":
               self.optimizer = SGD(self.lr_rate)
          print("optimzier : " + optimizer_name)

     def training(self, savepoint1, savepoint2, savepoint3, pickle_name, graph_option = True, print_step = True, save = True):
          savepk = save_load_parameter("./logs" + "/" + pickle_name)
          train_size = self.data["train_data"].shape[0]
          itr = int(train_size/self.batch_size)
          pre_acc = 0
          for i in range(self.epoch):
               for j in range(itr):
                    startT = time.time()
     
                    batch_mask = self.batch_size
                    train_data = self.data["train_data"][j*batch_mask:(j+1)*batch_mask]
                    train_label = self.data["train_label"][j*batch_mask:(j+1)*batch_mask]

                    grads = self.network.gradient(train_data, train_label)

                    self.optimizer.update(grads, self.network)
                  
                    #train_loss = self.network.loss(train_data, train_label)
                    train_loss = self.network.loss(train_data, train_label)
                    self.train_loss_list.append(train_loss)
                    if print_step == True:
                         print("step "+ str(j) + " train loss : " + str(train_loss))
                    if train_loss < 0.5:
                         if save == True:
                              savepk.save_parameter(self.network, pickle_name + "_better_loss.pkl")
                    
                    if j % savepoint1 == 1:
                         print("============================")
                         print("epoch : " + str(i+1))
                         print("itr : " + str(j+1))
                         finT = time.time()
                         if j != 0:
                              print("time : " + str((finT - startT)*savepoint1))
                         train_acc = self.network.accuracy(train_data, train_label)
                         if train_acc > pre_acc:
                              if save == True:
                                   savepk.save_parameter(self.network, pickle_name + "_better_accuracy.pkl")
                                   pre_acc = train_acc
                              
                         self.train_acc_list.append(train_acc)
                         print("============================")
                         print("train acc : " + str(train_acc))
                         #print("train loss : " + str(train_loss))
                         print("============================")
                         
                    if j != 0 and j % savepoint2 == 0:
                         valid_data = self.data["valid_data"]
                         valid_label = self.data["valid_label"]

                         valid_acc = self.network.accuracy(valid_data, valid_label)
                         self.valid_acc_list.append(valid_acc)
                         
                         print("valid_acc : " + str(valid_acc))
                         print("============================")
                         if save == True:
                              savepk.save_parameter(self.network, pickle_name + "_parameter " + str(i+1) + "-" + str(j+1) + ".pkl")
                         
                    if j != 0 and j % savepoint3 == 0:
                         valid_acc = self.network.accuracy(valid_data, valid_label)
                         self.valid_acc_list.append(valid_acc)
                         
                         print("valid_acc : " + str(valid_acc))
                         print("============================")
                         if save == True:
                              savepk.save_parameter(self.network, pickle_name + "_parameter " + str(i+1) + "-" + str(j+1) + ".pkl")
                              
          if save == True:
               savepk.save_parameter(self.network, pickle_name + "_final.pkl")

          if graph_option == True:          
               graph(self.train_loss_list, self.train_acc_list, self.valid_acc_list)



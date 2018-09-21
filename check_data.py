from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os, sys

from get_cifar10 import save_load_cifar10
from get_myData import save_load_mydata, split_train_valid, data2arr
from get_dog_vs_cat import save_load_dog_cat

from trainNet import *
from myLayerNet import *
from Function.save_load_parameter import *

def random_choice_from_data(data, n):
     datas = data["valid_data"]

     maxN = int(datas.shape[0])
     print("valid data size : " + str(datas.shape))

     choice = np.random.choice(maxN, n, replace=False)

     return datas[choice]

def batching_data(data, path, n):
     if path == "random":
          batch_data = random_choice_from_data(data, n)
     else:
          batch_data = data2arr("./" + path)

     return batch_data
     

def check_cifar(n, data, network, path = "random"):
     batch_data = batching_data(data, path, n)
     
     label_list = ["airplane", "automobile" ,"bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

     prediction_label = network.checking(batch_data)

     plot(batch_data, prediction_label, label_list, 10)

def check_mydata(n, data, network, path = "random"):
     batch_data = batching_data(data, path, n)
     
     label_list = ["ILIN", "IU", "SUL", "SUZI"]

     prediction_label = network.checking(batch_data)

     plot(batch_data, prediction_label, label_list, 4)

def check_dog_cat(n, data, network, path = "random"):
     batch_data = batching_data(data, path, n)
     
     label_list = ["Cat", "Dog"]

     prediction_label = network.checking(batch_data)

     plot(batch_data, prediction_label, label_list, 2)


def plot(batch_data, predict, label_list, n_predict):
     n_batch = batch_data.shape[0]
     if n_batch > 8:
          width = 8
          height = int(n_batch/width)*2
     else:
          width = n_batch
          height = 2
     
     index = range(n_predict)
     
     fig = plt.figure(figsize=(10,5))

     for i in range(n_batch):
          img = arr2img(batch_data[i])
          img_plot = fig.add_subplot(height, n_batch, (i*2)+1)
          img_plot.imshow(img)
          
          #pred = np.sort(predict[i])[::-1]
          pred = predict[i]
          
          label_plot = fig.add_subplot(height,n_batch,(i*2)+2)
          label_plot.barh(index, pred, tick_label=label_list)          

     plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.7, hspace=0.2)
     plt.show()
                    

def arr2img(data):
     data = data.transpose(1, 2, 0)
     img = Image.fromarray((data*255).astype('uint8'), 'RGB')
     return img


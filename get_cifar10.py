#from six.moves import cPickle as pickle
import pickle
import numpy as np
import os

def load_batch(filename):
     with open(filename, "rb") as f:
          datadict = pickle.load(f, encoding = "bytes")
          #for key, value in datadict.items():
          #     print(key)
          
          img = datadict[b'data']
          label = datadict[b'labels']
          data = img.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
          label = np.array(label)

          one_hot_label = np.zeros((label.shape[0], 10))
          for i in range(label.shape[0]):
               value = label[i]
               one_hot_label[i, value] = 1
          
          return data, one_hot_label
          

def load(data_dir):
     datas = []
     labels = []

     for i in range(1, 6):
          f = os.path.join(data_dir, "data_batch_%d" % (i, ))
          data, label = load_batch(f)
          datas.append(data)
          labels.append(label)
     tr_data = np.concatenate(datas)
     tr_labels = np.concatenate(labels)
     del data, label
     te_data, te_labels = load_batch(os.path.join(data_dir, "test_batch"))
     return tr_data, tr_labels, te_data, te_labels

def get_cifar10_data(data_dir, n_training=49000, n_validation=1000, n_test=1000):
     train_data, train_label, test_data, test_label = load(data_dir)

     # ~49000
     batch = range(n_training)
     train_data_batch = train_data[batch]
     train_label_batch = train_label[batch]
     
     # 49000~50000
     batch = range(n_training, n_training + n_test)
     valid_data_batch = train_data[batch]
     valid_label_batch = train_label[batch]
     
     # ~1000
     batch = range(n_test)
     test_data_batch = test_data[batch]
     test_label_batch = test_label[batch]

     tr_data = train_data_batch.astype(np.float64)
     var_data = valid_data_batch.astype(np.float64)
     te_data = test_data_batch.astype(np.float64)

     tr_data = tr_data.transpose(0, 3, 1, 2)
     var_data = var_data.transpose(0, 3, 1, 2)
     te_data = te_data.transpose(0, 3, 1, 2)

     mean = np.mean(tr_data, axis=0)
     std = np.std(tr_data)

     """
     tr_data -= mean
     var_data -= mean
     te_data -= mean
     """
     
     """
     tr_data /= std
     var_data /= std
     te_data /= std
     """
     
     tr_data /= 255
     var_data /= 255
     te_data /= 255
     
     return{
          'train_data': tr_data, 'train_label': train_label_batch,
          'valid_data': var_data, 'valid_label': valid_label_batch,
          'test_data': te_data, 'test_label': test_label_batch,
          'mean': mean, 'std': std
     }

def save_load_cifar10():
     pickle_name = "cifar10"
     path = "./Data/cifar-10"
     pickle_path = os.path.join(path, pickle_name)
     if not os.path.exists(pickle_path):
          print("create pickle file..")
          cifar = get_cifar10_data(path)
          with open(pickle_path, 'wb') as f:
               pickle.dump(cifar, f, pickle.HIGHEST_PROTOCOL)
          print("created pickle.")
     else:
          print("pickle file exist")
          with open(pickle_path, 'rb') as f:
               cifar = pickle.load(f)
               #print(cifar)

     return cifar
          


if __name__ == "__main__":
     save_load_cifar10()











     
     






     


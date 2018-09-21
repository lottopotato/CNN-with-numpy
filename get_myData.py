from PIL import Image
import pickle
import os
import numpy as np


def load_batch(path, name, label, n, n_of_class = 4):
     img_arr = np.zeros((n, 3, 32, 32))
     label_arr = np.zeros((n, n_of_class))

     _path = os.path.join(path, name)

     i = 0
     for img in os.listdir(_path):
          img_path = os.path.join(_path, img)

          if i >= n:break

          img_arr[i] += data2arr(img_path)
          label_arr[i, label] = 1
          i += 1

     return img_arr, label_arr

def split_data_test(img_arr, label_arr, n_train = 12000, n_test=500):
     train_data = np.zeros((n_train, 3, 32, 32))
     train_labels = np.zeros((n_train, 3))
     test_data = np.zeros((n_test, 3, 32, 32))
     test_labels = np.zeros((n_test, 3))

     train_data = img_arr[:n_train]
     train_labels = label_arr[:n_train]
     test_data = img_arr[n_train:n_train+n_test]
     test_labels = label_arr[n_train:n_train+n_test]

     return train_data, train_labels, test_data, test_labels
     
def data2arr(data):
     img = Image.open(data)
     img = img.resize((32,32))
     img = img.convert("RGB")
     data = np.asarray(img, dtype="int32").transpose(2, 0, 1)

     return data

def suffle(data, label, n):
     rand = np.random.choice(n, n, replace=False)
     data = data[rand]
     label = label[rand]

     return data, label

def data_append(dataarr, labelarr, data, label, st, fin):
     dataarr[st:fin] = data
     labelarr[st:fin] = label

     return dataarr, labelarr
 
def get_mydata(path = "./Data/myData_female4"):   
     """
     number of data per each class : 55
     """
     maxsize = 220
     n_of_class = 4
     n_train_per_C = 45
     n_test_per_C = 10
     
     n_train = int(n_train_per_C*n_of_class)
     n_test = int(n_test_per_C*n_of_class)

     n_each_C = int( n_train_per_C + n_test_per_C)

     ILIN = "1.ILIN/"
     IU = "2.IU/"
     SUL = "3.SUL/"
     SUZI = "4.SUZI/"
     
     data = {}

     ILIN_img, ILIN_labels = load_batch(path, ILIN, 0, n_each_C)
     IU_img, IU_labels = load_batch(path, IU, 1, n_each_C)
     SUL_img, SUL_labels = load_batch(path, SUL, 2, n_each_C)
     SUZI_img, SUZI_labels = load_batch(path, SUZI, 3, n_each_C)

     data = np.zeros((maxsize, 3, 32, 32))
     labels = np.zeros((maxsize, n_of_class))

     data, labels = data_append(data, labels, ILIN_img, ILIN_labels, 0, n_each_C)
     data, labels = data_append(data, labels, IU_img, IU_labels, n_each_C, int(n_each_C*2))
     data, labels = data_append(data, labels, SUL_img, SUL_labels, int(n_each_C*2), int(n_each_C*3))
     data, labels = data_append(data, labels, SUZI_img, SUZI_labels, int(n_each_C*3), int(n_each_C*4))
     
     data, labels = suffle(data, labels, maxsize)

     data /= 255.0
     
     data = {"data":data, "label":labels, "n_train":n_train_per_C, "n_test":n_test_per_C}

     return data

def split_train_valid(data, n_of_train, n_of_valid):
     n_of_data = n_of_train + n_of_valid

     mask = np.random.choice(n_of_data, n_of_data)

     temp_data = data["data"][mask]
     temp_label = data["label"][mask]

     train_data = temp_data[:n_of_train]
     train_label = temp_label[:n_of_train]
     valid_data = temp_data[n_of_train:n_of_data]
     valid_label = temp_label[n_of_train:n_of_data]

     return{"train_data":train_data, "train_label":train_label,
            "valid_data":valid_data, "valid_label":valid_label}

     

def save_load_mydata():
     pickle_name = "female4.pkl"
     path = "./Data/myData_female4"
     pickle_path = os.path.join(path, pickle_name)
     if not os.path.exists(pickle_path):
          print("create female4 pickle file")
          data = get_mydata()
          with open(pickle_path, "wb") as f:
               pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
               print("saved")
     else:
          print("pickle file exist")
          with open(pickle_path, "rb") as f:
               data = pickle.load(f)

     return data
          
     

if __name__ == "__main__":
     save_load_mydata()
                          
                          

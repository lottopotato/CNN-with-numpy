from PIL import Image
import pickle
import os
import numpy as np


def load_batch(path, name, label, n):
     img_arr = np.zeros((n, 3, 32, 32))
     label_arr = np.zeros((n, 2))

     i = 0
     for img in os.listdir(path):
          img_path = os.path.join(path, name + "." + str(i) + ".jpg")
          print("procesing .." + str(img_path))

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
 
def get_dog_cat(path = "./Data/dog-cat"):
     maxsize = 25000
     n_of_class = 2
     n_train_per_C = 12000
     n_test_per_C = 500
     
     n_train = int(n_train_per_C*n_of_class)
     n_test = int(n_test_per_C*n_of_class)

     dog = "dog"
     cat = "cat"
     
     data = {}

     dog_img, dog_labels = load_batch(path +"/img", dog, 0, int(n_train_per_C+n_test_per_C))
     cat_img, cat_labels = load_batch(path +"/img", cat, 1, int(n_train_per_C+n_test_per_C))

     train_data = np.zeros((n_train, 3, 32, 32))
     train_labels = np.zeros((n_train, n_of_class))
     test_data = np.zeros((n_test, 3, 32, 32))
     test_labels = np.zeros((n_test, n_of_class))

     train_data[:n_train_per_C], train_labels[:n_train_per_C], test_data[:n_test_per_C], test_labels[:n_test_per_C]= split_data_test(dog_img, dog_labels)
     train_data[n_train_per_C:n_train], train_labels[n_train_per_C:n_train], test_data[n_test_per_C:n_test], test_labels[n_test_per_C:n_test] = split_data_test(cat_img, cat_labels)
     
     train_data, train_labels = suffle(train_data, train_labels, n_train)
     test_data, test_labels = suffle(test_data, test_labels, n_test)

     train_data /= 255.0
     test_data /= 255.0
     
     data = {"train_data": train_data, "train_label":train_labels, "valid_data":test_data, 'valid_label': test_labels}

     return data

def save_load_dog_cat():
     pickle_name = "dog-cat.pkl"
     path = "./Data/dog-cat"
     pickle_path = os.path.join(path, pickle_name)
     if not os.path.exists(pickle_path):
          print("create Dog vs Cat pickle file")
          data = get_dog_cat()
          with open(pickle_path, "wb") as f:
               pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
               print("saved")
     else:
          print("pickle file exist")
          with open(pickle_path, "rb") as f:
               data = pickle.load(f)

     return data
     

if __name__ == "__main__":
     save_load_dog_cat()
          
                          
                          

import pickle
import os

class save_load_parameter:
     def __init__(self, path):
          self.path = path
          
     def save_parameter(self, network, pk_name):
          params = {}
          for i in range(1, network.layer_len+1):
               if(i<network.con_len+1):
                    params["W" + str(i)] = network.layers["C" + str(i)].W
                    params["b" + str(i)] = network.layers["C" + str(i)].b
               else:
                    params["W" + str(i)] = network.layers["fc" + str(i)].W
                    params["b" + str(i)] = network.layers["fc" + str(i)].b

                    
          pk_path = os.path.join(self.path, pk_name)
          if not os.path.exists(self.path):
               os.makedirs(self.path)
          with open(pk_path, "wb") as f:
               pickle.dump(params, f)
               print(pk_name + " saved")

     def load_parameter(self, network, pk_name):
          pk_path = os.path.join(self.path, pk_name)
          if os.path.exists(pk_path):
               with open(pk_path, "rb") as f:
                    para = pickle.load(f)
                    print("load pickle")
          else:
               print("no file found")

          params = {}
          for key, value in para.items():
               #print(key + " ")
               params[key] = value
               
          for i in range(1, network.layer_len):
               if(i<network.con_len+1):
                    network.layers['C'+str(i)].W = params['W'+str(i)]
                    network.layers['C'+str(i)].b = params['b'+str(i)]
               else:
                    network.layers['fc'+str(i)].W = params['W'+str(i)]
                    network.layers['fc'+str(i)].b = params['b'+str(i)]
          return network

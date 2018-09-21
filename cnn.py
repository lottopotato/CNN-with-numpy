import sys

from get_cifar10 import save_load_cifar10
from get_myData import save_load_mydata, split_train_valid
from get_dog_vs_cat import save_load_dog_cat

from trainNet import *
from myLayerNet import *
from Function.save_load_parameter import *

from check_data import *

class _option:
     def __init__(self, data = None, epoch = None, batch_size = None, pickle_name = None,
              use_previous_params = None, print_step = None, save = None,
              graph_option = None, check = None,
              optimizer_name = "Adam", lr_rate=0.001):
          
          self.data = data
          self.epoch = epoch
          self.batch_size = batch_size
          self.pickle_name = pickle_name
          
          self.use_previous_params = use_previous_params
          self.print_step = print_step
          self.save = save
          self.graph_option = graph_option 
          self.check = check

          self.optimizer_name = optimizer_name
          self.lr_rate = lr_rate
  
     def print_option(self):
          print("======= OPTION =========")
          print("data : " + str(self.data))
          print("epoch : " + str(self.epoch))
          print("batch size : " + str(self.batch_size))
          print("pickle name : " + str(self.pickle_name))
          print("use previous params : " + str(self.use_previous_params))
          print("print result per step : " + str(self.print_step))
          print("use save : " + str(self.save))
          print("use graph : " + str(self.graph_option)) 
          print("check option : " + str(self.check))
          print("learning rate : " + str(self.lr_rate))
          print("========================")

     def insert_option(self):
          data_input = input("1. data(cifar or mydata or dog_cat) : ")
          while(True):
               if data_input == "cifar" or data_input == "mydata" or data_input == "dog_cat":
                    break
               else:
                    data_input = input("retry, data : 1. cifar 2.mydata 3.dog_cat : ")
          self.data = data_input
          self.epoch = int(input("2. epoch : "))
          self.batch_size = int(input("3. batch size : "))
          self.pickle_name = input("4. pickle name : ")
          
          input5 = input("5. use previous params (True or T or t) or (False or F or f) ")
          while(True):
               if input5 == "True" or input5 == "T" or input5 == "t":
                    self.use_previous_params = True
                    break
               elif input5 == "False" or input5 == "F" or input5 == "f":
                    self.use_previous_params = False
                    break
               else:
                    input5 = input(" - input only (True or T or t) or (False or F or f) ")

          input6 = input("6. print_step (True or T or t) or (False or F or f) ")
          while(True):
               if input6 == "True" or input6 == "T" or input6 == "t":
                    self.print_step = True
                    break
               elif input6 == "False" or input6 == "F" or input6 == "f":
                    self.print_step = False
                    break
               else:
                    input6 = input(" - input only (True or T or t) or (False or F or f) " )

          input7 = input("7. save option (True or T or t) or (False or F or f) ")
          while(True):
               if input7 == "True" or input7 == "T" or input7 == "t":
                    self.save = True
                    break
               elif input7 == "Flase" or input7 == "F" or input7 == "f":
                    self.save = False
                    break
               else:
                    input7 = input(" - input only (True or T or t) or (False or F or f) " )

          input8 = input("8. graph option (True or T or t) or (False or F or f) ")
          while(True):
               if input8 == "True" or input8 == "T" or input8 == "t":
                    self.graph_option = True
                    break
               elif input8 == "False" or input8 == "F" or input8 == "f":
                    self.graph_option = False
                    break
               else:
                    input8 = input(" - input only (True or T or t) or (False or F or f) " )
               
          input9 = input("9. check option (True or T or t) or (False or F or f) ")
          while(True):
               if input9 == "True" or input9 == "T"  or input9 == "t":
                    self.check = True
                    break
               elif input9 == "False" or input9 == "F" or input9 == "f":
                    self.check = False
                    break
               else:
                    input9 = input(" - input only (True or T or t) or (False or F or f) " )
                                          
def train_cifar(option, n_class = 10, n_check = 6):
     """
     cifar 10
     class : 10
     number of image : 60000
     number of image per each class : 6000
     use data : 50000 , test data : 10000

     train data : 49000 , valid data : 1000
 
     """
     ## print option
     option.print_option()
     
     ## network
     network = myLayerNet(input_size = 32, n_class = n_class)
     ## data load
     cifar_data = save_load_cifar10()

     # use parameter from pickle stored previous training parameter
     if option.use_previous_params == True:
          savePk = save_load_parameter("./logs/cifar")
          network = savePk.load_parameter(network, "cifar_final.pkl")

     # training
     cifar_train = train(cifar_data, network, option.epoch, option.batch_size,
                          option.optimizer_name, option.lr_rate) 
     cifar_train.training(50, 200, 489, option.pickle_name,
                           option.graph_option, option.print_step, option.save)

     # checking using pyplot
     if option.check == True:
          check_cifar(n_check, cifar_data, network)

def train_mydata(option, n_class = 4, n_check = 6):
     """
     mydata
     class : 4
     number of image : 220
     number of image per each class : 55

     number of train data : 180
     number of valid data : 40
     """
     ## print option
     option.print_option()
     
     ## network
     network = myLayerNet(input_size = 32, n_class = n_class)

     # use parameter from pickle stored previous training parameter
     if option.use_previous_params == True:          
         #mydata parameter
          savePk = save_load_parameter("./logs/mydata")
          network = savePk.load_parameter(network, "mydata_final.pkl")
          
     ## Iterative learning
     train_itr= 1
     acc_list = 0

     for i in range(train_itr):
          mydata = save_load_mydata()
          mydata = split_train_valid(mydata, 180, 40)
     
          mydata_train = train(mydata, network, option.epoch, option.batch_size,
                               option.optimizer_name, option.lr_rate) 
          mydata_train.training(5, 10, 17, option.pickle_name,
                                option.graph_option, option.print_step, option.save)

          acc_list += mydata_train.valid_acc_list[int(option.epoch*2)-1]
          
     print(str(train_itr) + " train acc mean : " + str(acc_list/train_itr))

     ## check 
     if option.check == True:
          check_mydata(n_check, mydata, network)

def train_dog_cat(option, n_class = 2, n_check = 6):
     """
     dog vs cat
     class : 2
     number of image : 25000
     number of image per each class : 12500

     number of train data : 24000
     number of valid data : 1000
     """
     ## print option
     option.print_option()
     
     ## network
     network = myLayerNet(input_size = 32, n_class = n_class)
     ## data load
     dog_cat = save_load_dog_cat()

     # use parameter from pickle stored previous training parameter
     if option.use_previous_params == True:
          savePk = save_load_parameter("./logs/dog_cat")
          network = savePk.load_parameter(network, "dog_cat_final.pkl")
     
     dog_cat_train = train(dog_cat, network, option.epoch, option.batch_size,
                          option.optimizer_name, option.lr_rate) 
     dog_cat_train.training(50, 100, 239, option.pickle_name,
                           option.graph_option, option.print_step, option.save)
     # check
     if option.check == True:
          check_dog_cat(n_check, dog_cat, network)

def train_custom_setting():
     print("\n\
          ================================== 제약 조건 ==================================== \n\
           data : 데이터의 이름은 cifar, mydata, dog_cat 중 하나여야 합니다.\n\
           batch size : 배치의수는 학습용 데이터의 수량보다 적거나 같고 약수여야합니다.\n\
                          10이나 100이 적당합니다.\n\
          =================================  주요 옵션 ====================================\n\
           use_previous_params : 이전에 학습한 파라미터를 사용할지 결정합니다.\n\
           print_step : 학습과정 동안 스텝간의 loss나 accuracy를 화면에 출력할지 결정합니다.\n\
           save option : 세이브 포인트간 파라미터를 저장할지 결정합니다.\n\
           graph option : 전체 loss와 accuracy를 그래프로 표현할지 결정합니다.\n\
           check option : 검증데이터에서 샘플링한 데이터의 예측값을 표현합니다.\n\
          =================================================================================" )
          
     option = _option()
     option.insert_option()
     if option.data == "cifar":
          train_cifar(option)
     elif option.data == "mydata":
          train_mydata(option)
     elif option.data == "dog_cat":
          train_dog_cat(option)
     else:
          print("")
     
  
def main(argv):
     try:
          data_name = argv[1]
     
          if data_name == "cifar":
               ## option setting
               option = _option("cifar10", 1, 100, "cifar",
                                True, True, False, True, True)
               train_cifar(option)
          elif data_name == "mydata":
               ## option setting
               option = _option("female 4 classification", 1, 10, "mydata",
                                True, False, False, True, True)
               train_mydata(option)
          elif data_name == "dog_cat":
               ## option setting
               option = _option("dog vs cat", 1, 100, "dog_cat",
                                True, True, False, True, True)
               train_dog_cat(option)
          elif data_name == "setting":
               train_custom_setting()
     except IndexError:
          print("=============== USAGE =============")
          print(" python cnn.py cifar : train cifar")
          print(" python cnn.py mydat : train self data")
          print(" python cnn.py dog_cat : train dog vs cat")
          print(" python cnn.py setting : train custom setting")
          print(" =========== TRY AGAIN ===========")

if __name__ == "__main__":
     main(sys.argv)
     

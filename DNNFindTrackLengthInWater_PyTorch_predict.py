import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "../data_forRecoLength_04202019.csv"
#infile = "../data/data_forRecoLength_05202019.csv"
##infile = "data_forRecoLength_beamlikeEvts.csv"
infile = "data_for_trackLength_training.csv"
#infile = "../data/data_forRecoLength_06082019CC0pi.csv"
#infile = "../LocalFolder/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "../data_forRecoLength_04202019.csv"
#infile2 = "../data/data_forRecoLength_05202019.csv"
##infile2 = "data_forRecoLength_beamlikeEvts.csv"
infile2 = "data_for_trackLength_training.csv"
#infile2 = "../data/data_forRecoLength_06082019CC0pi.csv"
#infile2 = "../LocalFolder/NEWdata_forRecoLength_0_8MRD.csv"
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset = np.array(pd.read_csv(filein))
#Dataset1=np.delete(Dataset,obj=1398,axis=0)
np.random.shuffle(Dataset)
print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)

#--- events for predicting
filein2 = open(str(infile2))
print("events for prediction in: ",filein2)
Dataset2 = np.array(pd.read_csv(filein2))
#Dataset22=np.delete(Dataset2,obj=1398,axis=0)
np.random.seed(seed)
np.random.shuffle(Dataset2)
print(Dataset2)
features2, lambdamax2, labels2, rest2 = np.split(Dataset2,[2203,2204,2205],axis=1)
print( "lambdamax2 ", lambdamax2[:2], labels[:2])
print(features2[0])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]
test_x = features2[2000:]
test_y = labels2[2000:]
#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

# Custom Neural Network class
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2203, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 1)
        # self.fc1 = nn.Linear(2203, 100)
        # self.fc2 = nn.Linear(100, 10)
        # self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        
        # Initialize the weights
        # nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Instantiate the model
model = SimpleModel()

# Load weights from the file (if using the saved model weights)
model.load_state_dict(torch.load("weights_bets.pt"))

print("Created model and loaded weights from file")

# Set the model to evaluation mode
model.eval()

# Convert the test data to PyTorch tensors
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)

# Transform the test data using the same scaler used for training
x_transformed = torch.tensor(scaler.transform(test_x), dtype=torch.float32)

# Perform prediction
print('predicting...')
with torch.no_grad():
    y_predicted_tensor = model(x_transformed)

# Convert the predictions back to numpy array
y_predicted = y_predicted_tensor.numpy()

# Calculate metrics using sklearn
scores = metrics.mean_squared_error(y_predicted, test_y)
print("MSE (sklearn): {:.2f}".format(scores))

# Save results to a CSV file
print("Saving .csv file with energy variables...")
print("shapes: ", test_y.shape, ", ", y_predicted.shape)

df = pd.DataFrame({'TrueTrackLengthInWater': test_y.reshape(-1), 'DNNRecoLength': y_predicted.reshape(-1)})

#---read .csv file containing predict
np.random.seed(seed)
filein2 = open(str(infile2))
df1=pd.read_csv(filein2)
Dataout=np.array(df1)
#Dataout1=np.delete(Dataout,obj=1398,axis=0)
np.random.shuffle(Dataout)
print(Dataout)
df0 = pd.DataFrame(Dataout[2000:],columns=df1.columns)
#    print(df0.head())
#    df0= pd.read_csv("../LocalFolder/data_forRecoLength_9.csv")
print("df0.shape: ",df0.shape," df.shape: ",df.shape)
print("df0.head(): ",df0.head())
print("df.head(): ", df.head())
#df_final = pd.concat([df0,df], axis=1).drop(['lambda_max.1'], axis=1)
df_final = df0
print("-- Prev: df_final.columns: ",df_final.columns)
df_final.insert(2217, 'TrueTrackLengthInWater', df['TrueTrackLengthInWater'].values, allow_duplicates="True")
df_final.insert(2218, 'DNNRecoLength', df['DNNRecoLength'].values, allow_duplicates="True")
#deleting two events with wrong reconstruction
DNNRecoLength=df_final['DNNRecoLength']
#This loop excludes any events with reconstructed length >1000 as it doesn't match the MC values
i=0
a=[]
for y in DNNRecoLength:
   if y>1000:
     print("RecoLength:",y,"Event:",i)
     a.append(i)
   i=i+1
df_final=df_final.drop(df_final.index[a])
print("df_final.head(): ",df_final.head())
print("--- df_final.shape: ",df_final.shape)
print("-- After: df_final.columns: ",df_final.columns)

#-logical tests:
print("checking..."," df0.shape[0]: ",df0.shape[0]," len(y_predicted): ", len(y_predicted))
assert(df0.shape[0]==len(y_predicted))
print("df_final.shape[0]: ",df_final.shape[0]," df.shape[0]: ",df.shape[0])
assert(df_final.shape[0]==df.shape[0]-len(a))

#df_final.to_csv("../LocalFolder/vars_Ereco.csv", float_format = '%.3f')
#df_final.to_csv("vars_Ereco_04202019.csv", float_format = '%.3f')
#df_final.to_csv("vars_Ereco_05202019.csv", float_format = '%.3f')
#df_final[:600].to_csv("vars_Ereco_train_05202019.csv", float_format = '%.3f') #to be used for the energy BDT training
#df_final[600:].to_csv("vars_Ereco_pred_05202019.csv", float_format = '%.3f') #to be used for the energy prediction

df_final.to_csv("vars_Ereco.csv", float_format = '%.3f')
df_final[:(1187-len(a))].to_csv("vars_Ereco_train.csv", float_format = '%.3f')
df_final[(1187-len(a)):].to_csv("vars_Ereco_pred.csv", float_format = '%.3f')

#df_final.to_csv("vars_Ereco_06082019CC0pi.csv", float_format = '%.3f')
#df_final[:1000].to_csv("vars_Ereco_train_06082019CC0pi.csv", float_format = '%.3f')
#df_final[1000:].to_csv("vars_Ereco_pred_06082019CC0pi.csv", float_format = '%.3f')

#---if asserts fails check dimensions with these print outs:
#print("df: ",df.head())
#print(df.iloc[:,2200:])
#print(df0.head())
#print(df0.shape)
#print(df0.iloc[:,2200:])
#print(df_final.shape)
#print(df_final.iloc[:,2200:])
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
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
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator


#--------- File with events for reconstruction:
#--- evts for training:
#infile = "../data_forRecoLength_04202019.csv"
#infile = "../data/data_forRecoLength_05202019.csv"
##infile = "data_forRecoLength_beamlikeEvts.csv"
infile = "data_for_trackLength_training.csv"
#infile = "../LocalFolder/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "../data_forRecoLength_04202019.csv"
#infile2 = "../data/data_forRecoLength_05202019.csv"
#infile2 = "../LocalFolder/NEWdata_forRecoLength_0_8MRD.csv"
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein))
print(Dataset)
np.random.shuffle(Dataset)
print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)

#--- events for predicting
#filein2 = open(str(infile2))
#print("events for prediction in: ",filein2)
#Dataset2 = np.array(pd.read_csv(filein2))
#features2, lambdamax2, labels2, rest2 = np.split(Dataset2,[2203,2204,2205],axis=1)
#print( "lambdamax2 ", lambdamax2[:2], labels[:2])
#print(features2[0])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]
#test_x = features2[1000:]
#test_y = labels2[1000:]
#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

# Define your model class
class CustomModel(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2, init_mode, activation):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons1)
        self.fc2 = nn.Linear(neurons1, neurons2)
        self.fc3 = nn.Linear(neurons2, 1)

        # Initialize the weights dynamically based on init_mode
        self.init_weights(init_mode)

        # Choose the activation function dynamically
        self.activation = self.get_activation(activation)

    def init_weights(self, init_mode):
        if init_mode == 'uniform':
            nn.init.uniform_(self.fc1.weight)
            nn.init.uniform_(self.fc2.weight)
            nn.init.uniform_(self.fc3.weight)
        elif init_mode == 'lecun_uniform':
            nn.init.lecun_uniform_(self.fc1.weight)
            nn.init.lecun_uniform_(self.fc2.weight)
            nn.init.lecun_uniform_(self.fc3.weight)
        elif init_mode == 'normal':
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            nn.init.normal_(self.fc3.weight)
        elif init_mode == 'zero':
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc3.weight)
        elif init_mode == 'glorot_normal':
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
        elif init_mode == 'glorot_uniform':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
        elif init_mode == 'he_normal':
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        elif init_mode == 'he_uniform':
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        else:
            raise ValueError("Unsupported init_mode.")

    def get_activation(self, activation):
        if activation == 'softmax':
            return nn.Softmax(dim=-1)
        elif activation == 'softplus':
            return nn.Softplus()
        elif activation == 'softsign':
            return nn.Softsign()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'hard_sigmoid':
            return nn.Hardsigmoid()
        elif activation == 'linear':
            return nn.Identity()
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x
    
# Create a PyTorch estimator
class PyTorchEstimator(BaseEstimator):
    def __init__(self, input_dim, neurons1=25, neurons2=25, init_mode='normal', activation='relu',
                 optimizer='Adam', epochs=10, batch_size=2, verbose=0):
        self.input_dim = input_dim
        self.neurons1 = neurons1
        self.neurons2 = neurons2
        self.init_mode = init_mode
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        # Create the model architecture
        self.model = CustomModel(input_dim=self.input_dim, neurons1=self.neurons1, neurons2=self.neurons2,
                                 init_mode=self.init_mode, activation=self.activation)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        loss_fn = nn.MSELoss()

        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        elif self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        elif self.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=0.01)
        elif self.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        elif self.optimizer == 'Adamax':
            optimizer = optim.Adamax(self.model.parameters(), lr=0.002)
        elif self.optimizer == 'Nadam':
            optimizer = optim.Nadam(self.model.parameters(), lr=0.002)
        else:
            raise ValueError("Unsupported optimizer.")

        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, X.size(0), self.batch_size):
                batch_x = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).numpy()
        return y_pred
    
# Grid search parameters
param_grid = {
    'neurons1': [25, 50, 70, 90, 100],
    'neurons2': [5, 10, 15, 20, 25, 30],
    'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
    'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    'epochs': [10, 12, 15],
    'batch_size': [1, 2, 5]
}

# Create the estimator
estimator = PyTorchEstimator(input_dim=2203)

# Perform the grid search
#grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)
# adjust n_iter parameter
grid = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_iter=300, n_jobs=-1, scoring='neg_mean_squared_error')
grid_result = grid.fit(train_x, train_y)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
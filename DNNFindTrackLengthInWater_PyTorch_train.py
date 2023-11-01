import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#--------- File with events for reconstruction:
#--- evts for training:
infile = "data_for_trackLength_training.csv"

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
Dataset=np.array(pd.read_csv(filein))
np.random.shuffle(Dataset)  #shuffling the data sample to avoid any bias in the training
#print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1) #labels = TrueTrackLengthInWater
#print(rest)
#print(features[:,2202])
#print(features[:,2201])
#print(labels)
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]

print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
#Best: -45753.141985 using {'optimizer': 'RMSprop', 'neurons2': 10, 'neurons1': 100, 'init_mode': 'he_uniform', 'epochs': 12, 'batch_size': 2, 'activation': 'sigmoid'}
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2203, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 1)
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

def create_model():
    model = SimpleModel()
    return model

criterion = nn.MSELoss()
learning_rate = 0.001
epochs = 10
batch_size = 2

# Initialize the model and optimizer
model = create_model()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate) # Adamax

train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    for i in range(0, len(train_x_tensor), batch_size):
        inputs = train_x_tensor[i:i + batch_size]
        targets = train_y_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average loss for the epoch
    train_loss = running_loss / (len(train_x_tensor) / batch_size)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_outputs = model(train_x_tensor)
        val_loss = criterion(val_outputs, train_y_tensor)
        val_losses.append(val_loss.item())

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the model checkpoint
torch.save(model.state_dict(), "weights_bets.pt")

plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='training loss')
plt.plot(range(1, epochs+1), val_losses, label='validation loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.legend()
plt.xlim(1, epochs)
plt.savefig("Pytorch/train_test.pdf")
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch

print("Imports done")

# set seed
seed = 42

with open('data.pkl', "rb") as f: 
    classes, features = pickle.load(f)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, classes, test_size = 0.25, random_state = seed)
#x_train, x_test, y_train, y_test = Dataset(x_train), Dataset(x_test), Dataset(y_train), Dataset(y_test)

x_train = torch.tensor(x_train.values.astype(np.float32))
x_test = torch.tensor(x_test.values.astype(np.float32))
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

train= data_utils.TensorDataset(x_train, y_train)
test = data_utils.TensorDataset(x_test, y_test)

train_loader = data_utils.DataLoader(train, batch_size= 10)

n_features = 13


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)

net = Net()

learning_rate = 0.01
# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# create a loss function, negative log likelihood
#criterion = nn.NLLLoss()

criterion = nn.CrossEntropyLoss()

# run the main training loop
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = Variable(data), Variable(target)
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        #data = data.view(-1, 28*28)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.data[0]))
        print(f"Epoch: {epoch}, batch: {batch_idx}, loss: {loss.data}")
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(data), len(train_loader.dataset),
        #                    100. * batch_idx / len(train_loader), loss.data.item()))
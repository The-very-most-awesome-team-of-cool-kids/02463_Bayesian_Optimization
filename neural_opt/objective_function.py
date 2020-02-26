from data_load import data_load
from torch_neural import *
import torch.nn as nn
import torch.optim as optim
import time
import os

# possible cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
trainloader, testloader, classes = data_load()

#optimizer dict
optimizer_dict = {0: "SGD", 1: "ADAM"}

def objective_function(x):
    save_path = './cifar_net.pth'
    criterion = nn.CrossEntropyLoss()
    
    params = x[0] 

    # initialize neural network
    net = Net()
    net = net.to(device)

    # define training parameters

    # learning rate
    learning_rate = params[0]

    #optimizer
    if params[1] == 0:
        optimizer = optim.SGD(net.parameters(), lr=float(learning_rate), momentum=0.9)
    elif params[1] == 1:
        optimizer = optim.Adam(net.parameters(), lr=float(learning_rate)) 

    # train model
    print("-"*30)
    print("Training started with parameters: learning rate: " + str(learning_rate) + ", optimizer: " + str(optimizer_dict[params[1]]))
    train_net(net, trainloader, criterion, optimizer, save_path)

    # test model
    accuracy = test_net(net, testloader, save_path)

    # remove saved model
    os.remove(save_path)


    return -accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

device =  'cuda' if torch.cuda.is_available() else 'cpu'
# Define neural net
class Net(nn.Module):
    
    def __init__(self):
        
        if device == 'cuda':
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5).cuda()
            self.pool = nn.MaxPool2d(2, 2).cuda()
            self.conv2 = nn.Conv2d(6, 16, 5).cuda()
            self.fc1 = nn.Linear(16 * 5 * 5, 120).cuda()
            self.fc2 = nn.Linear(120, 84).cuda()
            self.fc3 = nn.Linear(84, 10).cuda()
        else:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.to(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_net(neural_net, trainloader, criterion, optimizer, save_path):
    """
    Function to train neural network
    --------------------------------
    Parameters:

    neural_net: a torch neural network
    trainloader: a torch dataloader object with training data
    criterion: a torch loss function
    optimizer: a torch optimizer
    save_path: the path to save the model
    """
    t0 = time.time()
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = neural_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(neural_net.state_dict(), save_path)
    print(f'Finished training after {time.time()-t0} seconds')
    print(f'Model saved as {save_path}')



def test_net(neural_net, testloader, save_path):
    """
    Function to test neural network
    --------------------------------
    Parameters:

    neural_net: a torch neural network
    testloader: a torch dataloader object with test data
    save_path: the path to load the neural network model from
    """

    neural_net.load_state_dict(torch.load(save_path))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = neural_net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
    return accuracy


def make_prediction(neural_net, save_path, images, classes, p = True):
    """
    function to make prediction
    --------------------------
    parameters:

    neural_net: a torch neural network
    save_path: path to load neural network from
    images: images to predict class of
    classes: the possible labels
    p: whether to print result or not
    """

    neural_net.load_state_dict(torch.load(save_path))
    outputs = neural_net(images)

    _, predicted = torch.max(outputs, 1)
    if p:
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(len(images))))
    return predicted

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
from torch_neural import Net, train_net, imshow, make_prediction, test_net
from data_load import data_load
import torchvision
import torch

device =  'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader, classes = data_load()


dataiter = iter(testloader)
images, labels = dataiter.next()
#images, labels = images.to(device), labels.to(device)

# predictions and true labels
save_path = './cifar_net.pth'
net = Net().to(device)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
make_prediction(net, save_path, images, classes)
imshow(torchvision.utils.make_grid(images))


test_net(net, testloader, save_path)
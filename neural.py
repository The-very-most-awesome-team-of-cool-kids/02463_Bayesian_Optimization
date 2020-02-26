from torch import nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch.utils.data.TensorDataset as Dataset

# set seed
seed = 42

with open('data.pkl', "rb") as f: 
    classes, features = pickle.load(f)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, classes, test_size = 0.25, random_state = seed)
x_train, x_test, y_train, y_test = Dataset(x_train), Dataset(x_test), Dataset(y_train), Dataset(y_test)


# Hyperparameters for our network
input_size = 13
hidden_sizes = [10, 6]
output_size = 3

# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))




loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


learning_rate = 1e-4
for t in range(100):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x_train)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y_train)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad



print(y_pred)

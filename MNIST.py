import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from scipy.io import loadmat

mnist = loadmat("mnist-original.mat") # Dataset source : https://www.kaggle.com/datasets/avnishnish/mnist-original?resource=download
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]
print(mnist_data[0])
print(mnist_label)

class Linear_soft(nn.Module): # Defining linear model architecture
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self,x):
        x = self.softmax(self.layer(x))
        return x
    
LS = Linear_soft()

working_weights_mn = [[1. for i in range(0,len(mnist_data[0]))] for j in range(0,10)] # An equal weight for everything is a good starting point that will allow our log loss gradient descent to update all weights in all cases, by having always a non null probability that will make the log loss function not null.
coeff = np.array(working_weights_mn, dtype=np.float32) # creates a standard 1 weight for every input for all 10 neurons, based on uniform data structure length and number of neurons
biais = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.], dtype = np.float32) # 0 bias for each neuron
LS.layer.weight.data = torch.tensor(coeff)
LS.layer.bias.data = torch.tensor(biais)

summary(LS, input_size=(784,)) # Displays summary to be sure of what we do

entree = torch.tensor(mnist_data[0], dtype=torch.float32)
exit = LS(entree)
print("exit :", exit.tolist())
print(mnist_label[0]) # Initial run with all weights to 1

true_label = [0 for k in range(0,10)]
true_label[int(mnist_label[0])] = 1 # For log loss, converts scalar true label into signal with 0 or 1 if said image belongs to that class
learning_rate = 0.1 # base learning rate for the rest of the program

# the real deal : planning a loop to execute a certain amount of times to reach the optimum ?
with torch.no_grad(): # Mandatory else it doesn't let me apply manually the update
    for i in range(0,len(working_weights_mn)): # for each neuron
        for j in range(0,len(working_weights_mn[0])): # for each among the 784 weights
            if np.float64(learning_rate*1/784*(exit[i] - true_label[i])*mnist_data[0][j]) != 0:
                print(np.float64(learning_rate*1/784*(exit[i] - true_label[i])*mnist_data[0][j]))
                working_weights_mn[i][j] = working_weights_mn[i][j] - np.float64(learning_rate*1/784*(exit[i] - true_label[i])*mnist_data[0][j]) # implemented using exercise 11-12 of TD3

# new forward pass
LS.layer.weight.data = torch.tensor(np.array(working_weights_mn, dtype=np.float32))
exit = LS(entree)
print("exit :", exit.tolist())
print(mnist_label[0])
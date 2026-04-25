import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from scipy.io import loadmat
from random import uniform

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

working_weights_mn = [[uniform(0,1) for i in range(0,len(mnist_data[0]))] for j in range(0,10)] # Pick as weight a random number between 0 and 1, because all equal weights creates a list of equiprobable probabilities
coeff = np.array(working_weights_mn, dtype=np.float32) # creates a standard 1 weight for every input for all 10 neurons, based on uniform data structure length and number of neurons
biais = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.], dtype = np.float32) # 0 bias for each neuron
LS.layer.weight.data = torch.tensor(coeff)
LS.layer.bias.data = torch.tensor(biais)

summary(LS, input_size=(784,))

entree = torch.tensor(mnist_data[0], dtype=torch.float32)
exit = LS(entree)
print("exit :", exit)
print(mnist_label[0])
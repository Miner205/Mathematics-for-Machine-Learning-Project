import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from scipy.io import loadmat
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0) # Sets parameters of those modules with the same random seed to ensure reproductibility

mnist = loadmat("mnist-original.mat") # Dataset source : https://www.kaggle.com/datasets/avnishnish/mnist-original?resource=download
mnist_data = mnist["data"].T.astype(np.float32)
mnist_label = mnist["label"][0].astype(np.int64)
mnist_data /= 255.0 # normalisation
indices = np.random.permutation(len(mnist_data)) # shuffles dataset and labels correctly and in a coherent manner
mnist_data = mnist_data[indices]
mnist_label = mnist_label[indices]
print(mnist_data[0])
print(mnist_label)

X_train, X_test = mnist_data[:60000], mnist_data[60000:] # Splitting time
y_train, y_test = mnist_label[:60000], mnist_label[60000:]


def func(pct, allvalues): # This is for piechart percentage display
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

class Linear_soft(nn.Module): # Defining linear model architecture
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self,x):
        x = self.layer(x) # We'll apply softmax later in our model, you'll see
        return x
    
LS = Linear_soft()
criterion = nn.CrossEntropyLoss() # Cross entropy loss including softmaxxing !
optimizer = optim.SGD(LS.parameters(), lr=0.1) # The built in optimizer with the model parameters and learning rate to speed up the process


train_dataset = TensorDataset( # Making our datasets time
    torch.tensor(X_train),
    torch.tensor(y_train)
)

test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)


"""working_weights_mn = [[1. for i in range(0,len(mnist_data[0]))] for j in range(0,10)] # An equal weight for everything is a good starting point that will allow our log loss gradient descent to update all weights in all cases, by having always a non null probability that will make the log loss function not null.
coeff = np.array(working_weights_mn, dtype=np.float32) # creates a standard 1 weight for every input for all 10 neurons, based on uniform data structure length and number of neurons
biais = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.], dtype = np.float32) # 0 bias for each neuron
LS.layer.weight.data = torch.tensor(coeff)
LS.layer.bias.data = torch.tensor(biais)""" # legacy manual code

summary(LS, input_size=(784,)) # Displays summary to be sure of what we do

epochs = 5

for epoch in range(epochs):
    LS.train()
    total_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad() # Resets the stored gradients we obtained at last iteration
        outputs = LS(images) # Applies model to our training dataset
        loss = criterion(outputs, labels) # Calculates log loss
        loss.backward() # Calculates log loss gradient, does not do backward propagation !
        optimizer.step() # Applies the gradient descent with log loss

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

"""entree = torch.tensor(mnist_data[0], dtype=torch.float32)
exit = LS(entree)
print("exit :", exit.tolist())
print(mnist_label[0]) # Initial run with all weights to 1

learning_rate = 0.1 # base learning rate for the rest of the program

for k in range(0,5): # To prevent overfitting, equal amount of gradient descent iteration over the whole training set (5 iterations per thing) -> epoch functioning
    for l in range(0,len(mnist_data)-10000): # Over each image of the training dataset
        true_label = [0 for k in range(0,10)] # Updates at each new iteration
        true_label[int(mnist_label[l])] = 1 # For log loss, converts scalar true label into signal with 0 or 1 if said image belongs to that class

        with torch.no_grad(): # Mandatory else it doesn't let me apply manually the update
            for i in range(0,len(working_weights_mn)): # for each neuron
                for j in range(0,len(working_weights_mn[0])): # for each among the 784 weights
                    working_weights_mn[i][j] = working_weights_mn[i][j] - np.float64(learning_rate*1/784*(exit[i] - true_label[i])*mnist_data[l][j]) # implemented using exercise 11-12 of TD3

        # new forward pass
        LS.layer.weight.data = torch.tensor(np.array(working_weights_mn, dtype=np.float32))
        entree = torch.tensor(mnist_data[l], dtype=torch.float32)
        exit = LS(entree) # removed prints for pure calculation, only prints when reaching threshold
        if l % 6000 == 0: # Set at 10% bar
            print("Model training at epoch " + str(k) + " trained at " + str((l / 6000)*10) + "%")

predicted = []
for i in range(60000, len(mnist_data)):
    entree = torch.tensor(mnist_data[i], dtype=torch.float32)
    exit = LS(entree)
    max_i = 0
    for j in range(0,len(exit)): # Looks for index of max probabilities in output array
        if exit[j] > exit[max_i]:
            max_i = j
    predicted.append(float(max_i))""" # Other legacy manual code, for demo purposes


LS.eval() # We go into test mode
predicted = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = LS(images) # Make the inputs go through the model
        preds = torch.argmax(outputs, dim=1) # Do the argmax, aka assign a label following highest softmax probability obtained
        predicted.extend(preds.numpy()) # Add predictions to our list
        true_labels.extend(labels.numpy()) # add the concerned true labels to our list


prediction_errors = [0,0,0,0,0,0,0,0,0,0] # Stores errors in array following the true label
wrong_ones = [] # Stores the images seen as wrong, to project them later
for i in range(0,len(predicted)):
    if predicted[i] != mnist_label[i+60000]:
        prediction_errors[int(mnist_label[i+60000])] += 1
        wrong_ones.append(mnist_data[i+60000])

print(prediction_errors)

fig, axs = plt.subplots(1,2)
axs[0].bar([0,1,2,3,4,5,6,7,8,9], prediction_errors)
axs[0].set_title("Misattribution of label following the true nature of a number")
axs[0].set_xlabel("True Class")
axs[0].set_ylabel("Amount of misattributed labels")
axs[1].pie([len(wrong_ones), 10000-len(wrong_ones)], labels=["Errors", "Correct guesses"], autopct= lambda pct: func(pct, [len(wrong_ones), 10000-len(wrong_ones)]))
axs[1].set_title("Error proportion Chart")
plt.show()
# générer un pyplot qui montre chiffres fautifs au choix ?
# try training with different number of epochs
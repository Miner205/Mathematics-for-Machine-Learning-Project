import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from scipy.io import loadmat
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from random import randint


def display_images(images, indices,  global_title, labels, labi, colonnes=5): # This will be useful later to print the wrong predictions
    n = len(indices)
    lignes = (n + colonnes - 1) // colonnes

    plt.figure(figsize=(2 * colonnes, 2 * lignes))
    plt.suptitle(global_title)

    for i, idx in enumerate(indices):
        image_784 = images[idx]
        image_28x28 = np.array(image_784).reshape(28, 28) # turns image into a proper 28x28 list of list

        plt.subplot(lignes, colonnes, i + 1)
        plt.imshow(image_28x28, cmap="gray_r")
        plt.title(f"Image {idx}; predicted {labels[0][labi][idx]}; true {labels[1][labi][idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


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

class MultiLayer_soft1(nn.Module): # Defining multi-layer model architecture with 1 hidden layer
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784,100) # 100 neurons choice because multiple of number of class, and allows more fine graining while reducing a bit the size of our network
        self.out_layer = nn.Linear(100,10)
        self.relu = nn.ReLU() # ReLU is according to course of the most used activation function, so why not ?

    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.out_layer(x)
        return x
    
class MultiLayer_soft2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784,100)
        self.l2 = nn.Linear(100,100)
        self.out_layer = nn.Linear(100,10)
        self.relu = nn.ReLU() # ReLU is according to course of the most used activation function, so why not ?

    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.out_layer(x)
        return x

    
LS = Linear_soft()
criterion = nn.CrossEntropyLoss() # Cross entropy loss including softmaxxing !
optimizer = optim.SGD(LS.parameters(), lr=0.1) # The built in optimizer with the model parameters and learning rate to speed up the process

MLS1 = MultiLayer_soft1()
criter1 = nn.CrossEntropyLoss() # Reasons : 1) Other loss functions flattent the tensor like huber loss, not adapted for multi class classification 2) softmax included already 3) accurate same condition comparison for our study between linear and several layers models
opt1 = optim.SGD(MLS1.parameters(), lr=0.1)

MLS2 = MultiLayer_soft2()
criter2 = nn.CrossEntropyLoss()
opt2 = optim.SGD(MLS2.parameters(), lr=0.1)

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
summary(MLS1, input_size=(784,))
summary(MLS2, input_size=(784,))

epochs = 5
losses = [[],[],[]]

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

    losses[0].append(total_loss)
    print(f"Linear System @ Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

for epoch in range(epochs):
    MLS1.train()
    total_loss = 0.0

    for images, labels in train_loader:
        opt1.zero_grad() # Resets the stored gradients we obtained at last iteration
        outputs = MLS1(images) # Applies model to our training dataset
        loss = criter1(outputs, labels) # Calculates huber loss
        loss.backward() # Calculates huber gradient, does not do backward propagation !
        opt1.step() # Applies the gradient descent

        total_loss += loss.item()

    losses[1].append(total_loss)
    print(f"Multi Layer with 1 hidden System @ Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

for epoch in range(epochs):
    MLS2.train()
    total_loss = 0.0

    for images, labels in train_loader:
        opt2.zero_grad() # Resets the stored gradients we obtained at last iteration
        outputs = MLS2(images) # Applies model to our training dataset
        loss = criter2(outputs, labels) # Calculates huber loss
        loss.backward() # Calculates huber gradient, does not do backward propagation !
        opt2.step() # Applies the gradient descent

        total_loss += loss.item()

    losses[2].append(total_loss)
    print(f"Multi Layer with 2 hidden System @ Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

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
wo_true = [[],[],[]] # Stores true label for each wrong prediction
wo_false = [[],[],[]] # Same but with predicted
for i in range(0,len(predicted)):
    if predicted[i] != mnist_label[i+60000]:
        prediction_errors[int(mnist_label[i+60000])] += 1
        wrong_ones.append(mnist_data[i+60000])
        wo_true[0].append(mnist_label[i+60000])
        wo_false[0].append(predicted[i])

print(prediction_errors)

plt.figure(0) # Create separate window for each figure
fig, axs = plt.subplots(1,2)
axs[0].bar([0,1,2,3,4,5,6,7,8,9], prediction_errors)
axs[0].set_title("Misattribution of label following the true nature of a number [LS]")
axs[0].set_xlabel("True Class")
axs[0].set_ylabel("Amount of misattributed labels")
axs[1].pie([len(wrong_ones), 10000-len(wrong_ones)], labels=["Errors", "Correct guesses"], autopct= lambda pct: func(pct, [len(wrong_ones), 10000-len(wrong_ones)]))
axs[1].set_title("Error proportion Chart [LS]")
# plt.show()

# Test time for Multi layer with 1 hidden

MLS1.eval() # We go into test mode
predicted = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = MLS1(images) # Make the inputs go through the model
        preds = torch.argmax(outputs, dim=1) # Do the argmax, aka assign a label following highest softmax probability obtained
        predicted.extend(preds.numpy()) # Add predictions to our list
        true_labels.extend(labels.numpy()) # add the concerned true labels to our list


prediction_errors = [0,0,0,0,0,0,0,0,0,0] # Stores errors in array following the true label
wrong_ones_mls1 = [] # Stores the images seen as wrong, to project them later
for i in range(0,len(predicted)):
    if predicted[i] != mnist_label[i+60000]:
        prediction_errors[int(mnist_label[i+60000])] += 1
        wrong_ones_mls1.append(mnist_data[i+60000])
        wo_true[1].append(mnist_label[i+60000])
        wo_false[1].append(predicted[i])

print(prediction_errors)

plt.figure(1)
fig, axs = plt.subplots(1,2)
axs[0].bar([0,1,2,3,4,5,6,7,8,9], prediction_errors)
axs[0].set_title("Misattribution of label following the true nature of a number [MLS1]")
axs[0].set_xlabel("True Class")
axs[0].set_ylabel("Amount of misattributed labels")
axs[1].pie([len(wrong_ones_mls1), 10000-len(wrong_ones_mls1)], labels=["Errors", "Correct guesses"], autopct= lambda pct: func(pct, [len(wrong_ones), 10000-len(wrong_ones)]))
axs[1].set_title("Error proportion Chart [MLS1]")
# plt.show()

# MLS2 test

MLS2.eval() # We go into test mode
predicted = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = MLS2(images) # Make the inputs go through the model
        preds = torch.argmax(outputs, dim=1) # Do the argmax, aka assign a label following highest softmax probability obtained
        predicted.extend(preds.numpy()) # Add predictions to our list
        true_labels.extend(labels.numpy()) # add the concerned true labels to our list


prediction_errors = [0,0,0,0,0,0,0,0,0,0] # Stores errors in array following the true label
wrong_ones_mls2 = [] # Stores the images seen as wrong, to project them later
for i in range(0,len(predicted)):
    if predicted[i] != mnist_label[i+60000]:
        prediction_errors[int(mnist_label[i+60000])] += 1
        wrong_ones_mls2.append(mnist_data[i+60000])
        wo_true[2].append(mnist_label[i+60000])
        wo_false[2].append(predicted[i])

print(prediction_errors)

plt.figure(2)
fig, axs = plt.subplots(1,2)
axs[0].bar([0,1,2,3,4,5,6,7,8,9], prediction_errors)
axs[0].set_title("Misattribution of label following the true nature of a number [MLS2]")
axs[0].set_xlabel("True Class")
axs[0].set_ylabel("Amount of misattributed labels")
axs[1].pie([len(wrong_ones_mls2), 10000-len(wrong_ones_mls2)], labels=["Errors", "Correct guesses"], autopct= lambda pct: func(pct, [len(wrong_ones), 10000-len(wrong_ones)]))
axs[1].set_title("Error proportion Chart [MLS2]")

# Doing more graphs here for the studies

# comparing similarities between falsely labeled ones

mls1_ls_count = 0
mls2_ls_count = 0
print(type(wrong_ones))
print(type(wrong_ones_mls1))
print(type(wrong_ones_mls2))
print(type(wrong_ones[0]))
print(type(wrong_ones_mls1[0]))
print(type(wrong_ones_mls2[0]))
for elem in wrong_ones:
    if any(np.array_equal(elem, x) for x in wrong_ones_mls1):
        mls1_ls_count += 1
    if any(np.array_equal(elem, x) for x in wrong_ones_mls2):
        mls2_ls_count += 1

mls1_ls_count /= len(wrong_ones) # Get a proportion relatives to the length of LS incorrect predictions, for better understanding
mls2_ls_count /= len(wrong_ones)

plt.figure(3)
fig, axs = plt.subplots(1,2) # Loss progression for each combined in one graph + wrong ones similarities
axs[0].plot(np.arange(0,epochs,1), losses[0], linestyle="dashdot", color="blue", marker="+")
axs[0].plot(np.arange(0,epochs,1), losses[1], linestyle="dashdot", color="green", marker="+")
axs[0].plot(np.arange(0,epochs,1), losses[2], linestyle="dashdot", color="red", marker="+")
axs[0].set_title("Loss progression during epochs")
axs[0].set_ylabel("Total loss (u.a.)")
axs[0].set_xlabel("Epoch number")
axs[0].legend(["LS", "MLS1", "MLS2"])
axs[0].set_xticks(np.arange(0,epochs,1)) # Show every epoch in x axis
axs[1].bar(np.arange(0,len(["MultiLayer Softmax 1", "MultiLayer Softmax 2"])) - 0.35/2, [mls1_ls_count, mls2_ls_count], width=0.4)
axs[1].bar(np.arange(0,len(["MultiLayer Softmax 1", "MultiLayer Softmax 2"])) + 0.35/2, [len(wrong_ones_mls1)/len(wrong_ones), len(wrong_ones_mls2)/len(wrong_ones)], color="olivedrab", width=0.4) # Adds in comparison length of wrong ones array to LS wrong ones to make a relative comparison
axs[1].set_xticks(np.arange(0,len(["MultiLayer Softmax 1", "MultiLayer Softmax 2"])), ["MultiLayer Softmax 1", "MultiLayer Softmax 2"])
axs[1].legend(["Similar incorrect predictions", "Size of incorrect predictions array"])
axs[1].set_title("Proportion of incorrect predictions of Linear System similar to other models")
axs[1].set_ylim(0,1)
plt.show()

random_indexes = [randint(0, len(wrong_ones)) for _ in range(25)] # 25 random indexes to just have a glance, but not too much else matplotlib displays a mess
display_images(wrong_ones, random_indexes, "Selected random sample of wrong predictions from LS model", [wo_false, wo_true], 0)
random_indexes = [randint(0, len(wrong_ones_mls1)) for _ in range(25)]
display_images(wrong_ones_mls1, random_indexes, "Selected random sample of wrong predictions from MLS1 model", [wo_false, wo_true], 1)
random_indexes = [randint(0, len(wrong_ones_mls2)) for _ in range(25)]
display_images(wrong_ones_mls2, random_indexes, "Selected random sample of wrong predictions from MLS2 model", [wo_false, wo_true], 2)

# print the damn numbers
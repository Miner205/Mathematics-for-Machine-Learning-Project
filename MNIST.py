from scipy.io import loadmat
mnist = loadmat("mnist-original.mat") # Dataset source : https://www.kaggle.com/datasets/avnishnish/mnist-original?resource=download
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]
print(mnist_data[0])
print(mnist_label)
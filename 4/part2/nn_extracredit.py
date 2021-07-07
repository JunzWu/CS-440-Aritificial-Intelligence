# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from nn_main import plot_confusion_matrix
import os 

class ConvNet(nn.Module):
    def __init__(self):
        """ Initialize the layers of your neural network

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by 
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 4)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 10)
        
    def forward(self, xb):
        """ A forward pass of your neural network

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs
        @param xb: an (N, 8, 8) torch tensor
        @return: an (N, 10) torch tensor
        """
        N, W, H = xb.shape
        xb = xb.view(N, 1, W, H)
            
        out = F.relu(self.conv1(xb))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

def fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=1):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, containing the mean loss at the beginning of training and after each epoch
    """
    net.cuda()
    
    net.eval() #put the net in evaluation mode
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    val_dl = torch.utils.data.DataLoader(val)
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        losses, nums = zip(*[loss_batch(net, loss_func, X.cuda(), Y.cuda()) for X, Y in train_dl])
        train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        # TODO compute the validation loss and store it in a list
        losses, nums = zip(*[loss_batch(net, loss_func, X.cuda(), Y.cuda()) for X, Y in val_dl])
        val_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        
    for _ in range(n_epochs):
        net.train() #put the net in train mode
        # TODO 
        losses, nums=zip(*[loss_batch(net, loss_func, X.cuda(), Y.cuda(),optimizer) for X, Y in train_dl])
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            # TODO compute the train and validation losses and store it in a list
            losses_t, nums_t = zip(*[loss_batch(net, loss_func, X.cuda(), Y.cuda()) for X, Y in train_dl])
            loss_t=np.sum(np.multiply(losses_t, nums_t)) / np.sum(nums_t)
            train_epoch_loss.append(loss_t)
            losses_v, nums_v = zip(*[loss_batch(net, loss_func, X.cuda(), Y.cuda()) for X, Y in val_dl])
            loss_v=np.sum(np.multiply(losses_v, nums_v)) / np.sum(nums_v)
            val_epoch_loss.append(loss_v)
    return train_epoch_loss, val_epoch_loss

def loss_batch(model, loss_func, xb, yb, opt=None):
    """ Compute the loss of the model on a batch of data, or do a step of optimization.

    @param model: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer.  If not None, use the Optimizer to improve the model. Otherwise, just compute the loss.
    @return a numpy array of the loss of the minibatch, and the length of the minibatch
    """
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def accuracy(y_pre, y, num_classes):
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    
    n = y.shape[0]
    
    for i in range(n):
        if y_pre[i] == y[i]:
            avg_class_rate += 1.0
    avg_class_rate = avg_class_rate/n
    
    for i in range(num_classes):
        X = np.where(y==i)[0]
        for j in X:
            if y_pre[j] == y[j]:
                class_rate_per_class[i] += 1.0
        class_rate_per_class[i] = class_rate_per_class[i]/len(X)
    
    return avg_class_rate, class_rate_per_class

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    
    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")
    x_train = torch.tensor(np.reshape(x_train, [-1, 28, 28]), dtype=torch.float)
    y_train = torch.tensor(np.reshape(y_train, [-1]), dtype=torch.long)
    
    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")
    x_test = torch.tensor(np.reshape(x_test, [-1, 28, 28]), dtype=torch.float)
    y_test = torch.tensor(np.reshape(y_test, [-1]), dtype=torch.long)
    
    train = torch.utils.data.TensorDataset(x_train, y_train)
    val = torch.utils.data.TensorDataset(x_test, y_test)
    
    
    net=ConvNet()
    loss_fn = nn.CrossEntropyLoss()
    sgd=torch.optim.SGD(net.parameters(),lr=0.008)
    aa = torch.optim.lr_scheduler.ExponentialLR(sgd, 0.95)
    
    train_el1 = []
    val_el1 = []
    for i in range(50):
        print(i)
        train_el,val_el=fit_and_validate(net,sgd,loss_fn,train,val,1, batch_size = 100)
        aa.step()
        if i == 0:
            train_el1 += [train_el[0]]
            train_el1 += [train_el[1]]
            val_el1 += [val_el[0]]
            val_el1 += [val_el[1]]
        if i != 0:
            train_el1 += [train_el[1]]
            val_el1 += [val_el[1]]
            
    print(train_el1,val_el1)
    
    x=np.arange(51)
    plt.plot(x,train_el1,'r',label='train_epoch_loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.show()
    
    plt.plot(x,val_el1,'b',label='val_epoch_loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.show()
    
    torch.save(net.gpu().state_dict(), "conv.pb")
    #net.load_state_dict(torch.load("conv.pb"))
    
    
    y_train_pre = net.forward(x_train)
    y_train_pre = y_train_pre.detach()
    y_train_pre = y_train_pre.numpy()
    y_train_pre = np.argmax(y_train_pre, axis = 1)
    
    y_test_pre = net.forward(x_test)
    y_test_pre = y_test_pre.detach()
    y_test_pre = y_test_pre.numpy()
    y_test_pre = np.argmax(y_test_pre, axis = 1)
    
    y_train = y_train.numpy()
    y_test = y_test.numpy()
    
    avg_class_rate_train, class_rate_per_class_train = accuracy(y_train_pre, y_train, 10)
    print(avg_class_rate_train, class_rate_per_class_train)
    
    avg_class_rate_test, class_rate_per_class_test = accuracy(y_test_pre, y_test, 10)
    print(avg_class_rate_test, class_rate_per_class_test)
    
    class_names = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress",
                            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
    plot_confusion_matrix(y_train, y_train_pre, classes=class_names,
                          normalize=True, title='Confusion matrix for training data')
    plot_confusion_matrix(y_test, y_test_pre, classes=class_names,
                          normalize=True, title='Confusion matrix for test data')
    plt.show()

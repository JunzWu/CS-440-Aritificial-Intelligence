# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:32:09 2019

@author: Junz
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from nn_main import plot_confusion_matrix
import os
import CNN

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

net=CNN.ConvNet()
loss_fn = nn.CrossEntropyLoss()
sgd=torch.optim.SGD(net.parameters(),lr=0.005)
aa = torch.optim.lr_scheduler.ExponentialLR(sgd, 0.95)

train_el1 = []
val_el1 = []
for i in range(50):
    train_el,val_el=CNN.fit_and_validate(net,sgd,loss_fn,train,val,1, batch_size = 100)
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

x=np.arange(31)
plt.plot(x,train_el1,'r',label='train_epoch_loss')
plt.plot(x,val_el1,'b',label='val_epoch_loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.show()

torch.save(net.gpu().state_dict(), "conv.pb")

y_train_pre = net.forward(x_train)
y_train_pre = y_train_pre.numpy()
y_train_pre = np.argmax(y_train_pre, axis = 1)

y_test_pre = net.forward(x_test)
y_test_pre = y_train_pre.numpy()
y_test_pre = np.argmax(y_test_pre, axis = 1)

y_train = y_train.numpy()
y_test = y_test.numpy()

avg_class_rate_train, class_rate_per_class_train = CNN.accuracy(y_train_pre, y_train, 10)
print(avg_class_rate_train, class_rate_per_class_train)

avg_class_rate_test, class_rate_per_class_test = CNN.accuracy(y_test_pre, y_test, 10)
print(avg_class_rate_test, class_rate_per_class_test)

class_names = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress",
                        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
plot_confusion_matrix(y_train, y_train_pre, classes=class_names,
                      normalize=True, title='Confusion matrix for training data')
plot_confusion_matrix(y_test, y_test_pre, classes=class_names,
                      normalize=True, title='Confusion matrix for test data')
plt.show
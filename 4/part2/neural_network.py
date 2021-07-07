import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):

    #IMPLEMENT HERE
    losses =[]
    for e in range(epoch):
        if shuffle == True:
            index = [i for i in range(x_train.shape[0])]
            np.random.shuffle(index)
            x_train = x_train[index,:]
            y_train = y_train[index]
        loss = 0
        for i in range(int(x_train.shape[0]/200)):
            x = x_train[i*200:(i+1)*200,:]
            y = y_train[i*200:(i+1)*200]
            loss += four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x, y, False)
        losses += [loss]   
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    
    classification = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, True)
    n = y_test.shape[0]
    
    for i in range(n):
        if classification[i] == y_test[i]:
            avg_class_rate += 1.0
    avg_class_rate = avg_class_rate/n
    
    for i in range(num_classes):
        X = np.where(y_test==i)[0]
        for j in X:
            if classification[j] == y_test[j]:
                class_rate_per_class[i] += 1.0
        class_rate_per_class[i] = class_rate_per_class[i]/len(X)
    
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x, y, test):
    Z1, acache1 = affine_forward(x, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, w2, b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, w3, b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3, w4, b4)
    
    if test == True:
        classification = np.argmax(F, axis = 1)
        return classification
    loss, dF = cross_entropy(F, y)
    dA3, dw4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dw3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dw2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dx, dw1, db1 = affine_backward(dZ1, acache1)
    
    w1 -= 0.1*dw1
    w2 -= 0.1*dw2
    w3 -= 0.1*dw3
    w4 -= 0.1*dw4
    
    return loss

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    cache = (A, W, b)
    Z = np.dot(A, W)
    Z = Z+b
    return Z, cache

def affine_backward(dZ, cache):
    A, W, b = cache
    
    WT = W.T
    dA = np.dot(dZ, WT)
    
    AT = A.T
    dW = np.dot(AT, dZ)
    
    dB = np.sum(dZ, axis = 0)
    return dA, dW, dB

def relu_forward(Z):
    cache = Z
    A = np.maximum(Z,0)
    
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    X = np.where(Z<=0)
    dZ = dA
    dZ[X] = 0
    return dZ

def cross_entropy(F, y):
    n = F.shape[0]
    C = F.shape[1]
    loss = 0 
    for i in range(n):
        loss += F[i,int(y[i])]
        e = np.exp(F[i,:])
        e = np.sum(e)
        loss -= np.log(e)
    loss = -1*loss/n
    
    dF = np.zeros((n,C))
    for i in range(n):
        for j in range(C):
            if j == y[i]:
                dF[i,j] += 1
            e = np.exp(F[i,:])
            e = np.sum(e)
            dF[i,j] -= np.exp(F[i,j])/e
            dF[i,j] = -1*dF[i,j]/n
    return loss, dF

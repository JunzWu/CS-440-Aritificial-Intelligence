import numpy as np
from tqdm import tqdm


class MultiClassPerceptron(object):
    def __init__(self, num_class, feature_dim):
        """Initialize a multi class perceptron model.

        This function will initialize a feature_dim weight vector,
        for each class.

        The LAST index of feature_dim is assumed to be the bias term,
                self.w[:,0] = [w1,w2,w3...,BIAS]
                where wi corresponds to each feature dimension,
                0 corresponds to class 0.

        Args:
            num_class(int): number of classes to classify
            feature_dim(int): feature dimension for each example
        """

        self.w = np.zeros((feature_dim + 1, num_class))

    def train(self, train_set, train_label):
        """ Train perceptron model (self.w) with training dataset.

        Args:
            train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
            train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
        """

        # YOUR CODE HERE
        learning_rate = 0.01
        train_num = len(train_label)
        # initialize bias to 1
        train_set = np.hstack((train_set, np.ones((train_num, 1))))
        for idx, each_x in tqdm(enumerate(train_set), total=train_num, desc='PERCEPTRON MODEL TRAIN'):
            pred_y = np.argmax(each_x.dot(self.w))
            true_y = train_label[idx]
            if pred_y != true_y:
                self.w[:, true_y] += learning_rate * each_x
                self.w[:, pred_y] -= learning_rate * each_x

    def test(self, test_set, test_label):
        """ Test the trained perceptron model (self.w) using testing dataset. 
                The accuracy is computed as the average of correctness 
                by comparing between predicted label and true label. 

        Args:
            test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
            test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

        Returns:
                accuracy(float): average accuracy value 
                pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
        """

        # YOUR CODE HERE
        accuracy = 0
        test_num = len(test_label)
        test_set = np.hstack((test_set, np.ones((test_num, 1))))
        pred_label = np.argmax(test_set.dot(self.w), axis=1)
        accuracy = np.sum(pred_label == test_label) / len(pred_label)

        return accuracy, pred_label

    def save_model(self, weight_file):
        """ Save the trained model parameters 
        """

        np.save(weight_file, self.w)

    def load_model(self, weight_file):
        """ Load the trained model parameters 
        """

        self.w = np.load(weight_file)

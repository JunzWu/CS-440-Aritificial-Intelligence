import numpy as np
from tqdm import tqdm


class NaiveBayes(object):
    def __init__(self, num_class, feature_dim, num_value):
        """Initialize a naive bayes model.

        This function will initialize prior and likelihood, where
        prior is P(class) with a dimension of (# of class,)
                that estimates the empirical frequencies of different classes in the training set.
        likelihood is P(F_i = f | class) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class),
                that computes the probability of every pixel location i being value f for every class label.

        Args:
            num_class(int): number of classes to classify
            feature_dim(int): feature dimension for each example
            num_value(int): number of possible values for each pixel
        """

        self.num_value = num_value
        self.num_class = num_class
        self.feature_dim = feature_dim

        self.prior = np.zeros((num_class))
        self.likelihood = np.zeros((feature_dim, num_value, num_class))

    def train(self, train_set, train_label, k=1):
        """ Train naive bayes model (self.prior and self.likelihood) with training dataset.
                self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
                self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of
                        (# of features/pixels per image, # of possible values per pixel, # of class).
                You should apply Laplace smoothing to compute the likelihood.

        Args:
            train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
            train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
            # k Laplace smoothing parameter
        """

        # YOUR CODE HERE
        train_num = len(train_label)
        # estimate the priors P(class)
        for y in range(self.num_class):
            self.prior[y] = sum(train_label == y) / train_num
        # add k to numerator - initialize
        frequent_cnt = np.ones(
            shape=(self.feature_dim, self.num_value, self.num_class)) * k
        # set frequent_cnt by train data
        for X, y in tqdm(zip(train_set, train_label), total=len(train_label), desc="BAYES MODEL TRAIN"):
            for f_i, f in enumerate(X):
                frequent_cnt[f_i, f, y] += 1
        # set likeihood parameter
        for y in range(self.num_class):
            for f_i in range(self.feature_dim):
                self.likelihood[f_i, :, y] = frequent_cnt[f_i, :, y] / \
                    sum(frequent_cnt[f_i, :, y])

    def test(self, test_set, test_label):
        """ Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
                by performing maximum a posteriori (MAP) classification.
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
        pred_label = np.zeros((len(test_set)))
        probs = np.zeros((len(test_set)))
        # predict every sample X by likelihood
        for X_idx, X in tqdm(enumerate(test_set), total=len(pred_label), desc='BAYES MODEL TEST'):
            # initial final log_probs by prior prob
            # log_probs = self.prior.copy()
            log_probs = np.log(self.prior)
            for y_i in range(self.num_class):
                for f_i in range(self.feature_dim):
                    log_probs[y_i] += np.log(self.likelihood[f_i, X[f_i], y_i])
            this_predict_label = np.argmax(log_probs)
            pred_label[X_idx] = this_predict_label
            probs[X_idx]=max(log_probs)
        # calculate acc rate
        accuracy = np.sum(pred_label == test_label) / len(pred_label)

        return accuracy, pred_label, probs

    def save_model(self, prior, likelihood):
        """ Save the trained model parameters 
        """

        np.save(prior, self.prior)
        np.save(likelihood, self.likelihood)

    def load_model(self, prior, likelihood):
        """ Load the trained model parameters 
        """

        self.prior = np.load(prior)
        self.likelihood = np.load(likelihood)

    def intensity_feature_likelihoods(self, likelihood):
        """
        Get the feature likelihoods for high intensity pixels for each of the classes,
            by sum the probabilities of the top 128 intensities at each pixel location,
            sum k<-128:255 P(F_i = k | c).
            This helps generate visualization of trained likelihood images. 

        Args:
            likelihood(numpy.ndarray): likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class)
        Returns:
            feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
                (# of features/pixels per image, # of class)
        """
        # YOUR CODE HERE
        feature_likelihoods = np.sum(likelihood[:, 128:256, :], axis=1)

        return feature_likelihoods

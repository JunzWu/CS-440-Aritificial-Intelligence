# Main function for train, test and visualize.
# You do not need to modify this file.

import numpy as np
from perceptron import MultiClassPerceptron
from naive_bayes import NaiveBayes
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def load_dataset(data_dir=''):
    """Load the train and test examples
    """
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    return x_train, y_train, x_test, y_test


def plot_visualization(images, classes, cmap):
    """Plot the visualizations
    """
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax[i % 2, i // 2].imshow(images[:, i].reshape((28, 28)), cmap=cmap)
        ax[i % 2, i // 2].set_xticks([])
        ax[i % 2, i // 2].set_yticks([])
        ax[i % 2, i // 2].set_title(classes[i])
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # # ------------------add: display rate------------------
    # the classification rate for each fashion item
    for y, y_name in enumerate(classes):
        print('%s acc rate(class=%d): %.3f' % (y_name, y, cm[y, y]))
    # # ------------------add: display rate------------------
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label theam with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_samples(images, classes, class_names, cmap):
    """Plot the highest/lowest posterior probabilities samples"""
    fig, ax = plt.subplots(2, 10, figsize=(12, 5))
    for i in range(20):
        ax[i % 2, i // 2].imshow(images[i].reshape((28, 28)), cmap=cmap)
        ax[i % 2, i // 2].set_xticks([])
        ax[i % 2, i // 2].set_yticks([])
        this_class = classes[i // 2]
        flag = '(highest)' if i // 10 == 0 else '(lowest)'
        ax[i // 2, i % 2].set_title(class_names[this_class] + flag)
    plt.show()


if __name__ == '__main__':

    # Load dataset.
    x_train, y_train, x_test, y_test = load_dataset()
    # # ------------------add: sample test,faster------------------
    # x_train, y_train = x_train[:10000], y_train[:10000]
    # x_test, y_test = x_test[:500], y_test[:500]
    # # ------------------add: sample test,faster------------------
    # Initialize naive bayes model.
    num_class = len(np.unique(y_train))
    feature_dim = len(x_train[0])
    num_value = 256
    class_names = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress",
                            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])

    # ------------------add: best laplace smoothing parameter k
    # test_k, avg_acc_l = [0.1, 0.3, 1, 3, 9, 10], []
    # for laplace_k in test_k:
    #     NB = NaiveBayes(num_class, feature_dim, num_value)
    #     # Train model.
    #     NB.train(x_train, y_train, k=laplace_k)
    #     accuracy, _ = NB.test(x_test, y_test)
    #     avg_acc_l.append(accuracy)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('Figure: average_accuracy ~ k')
    # ax.set_xlabel('k')
    # ax.set_ylabel('average_accuracy')  # set coordinate axis
    # plt.plot(test_k, avg_acc_l, '.-')
    # plt.show()
    # ------------------add: best laplace smoothing parameter k
    NB = NaiveBayes(num_class, feature_dim, num_value)
    # Train model.
    NB.train(x_train, y_train)
    # Feature likelihood for high intensity pixels.
    feature_likelihoods = NB.intensity_feature_likelihoods(NB.likelihood)
    # Visualize the feature likelihoods for high intensity pixels.
    plot_visualization(feature_likelihoods, class_names, "Greys")
    # Classify the test sets.
    accuracy, y_pred, probs = NB.test(x_test, y_test)
    print('Bayes avg_acc: ', accuracy)

    # Plot confusion matrix.
    plot_confusion_matrix(y_test, y_pred, classes=class_names,
                          normalize=True, title='Confusion matrix, with normalization')
    plt.show()
    # ------------------add: plot highest/lowest posterior probabilities
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    diagnal_li = [cm[i, i] for i in range(num_class)]
    highest_class, lowest_class = np.argmax(diagnal_li), np.argmin(diagnal_li)
    tmp_arr1, tmp_arr2 = y_test == highest_class, y_test == lowest_class
    sample = np.zeros(20)
    for i in range(10):
        tmp_arr = y_test == i
        sample_idx = np.where(tmp_arr == True)[0]
        probs1 = probs[sample_idx]
        maximum = np.argmax(probs1)
        minimum = np.argmin(probs1)
        sample[2*i] = int(sample_idx[maximum])
        sample[2*i+1] = int(sample_idx[minimum])
    sample.astype(int)
    sample_x, sample_y = x_test[sample], y_test[sample]
    plot_samples(sample_x, sample_y, class_names, "Greys")
    '''
    sample_idx1 = np.random.choice(np.where(tmp_arr1 == True)[0], 5)
    sample_idx2 = np.random.choice(np.where(tmp_arr2 == True)[0], 5)
    sample_idx = np.hstack([sample_idx1, sample_idx2])
    sample_x, sample_y = x_test[sample_idx], y_test[sample_idx]
    plot_samples(sample_x, sample_y, class_names, "Greys")
    # ------------------add: plot highest/lowest posterior probabilities
    # Initialize perceptron model.
    '''
    '''
    perceptron = MultiClassPerceptron(num_class, feature_dim)
    # Train model.
    perceptron.train(x_train, y_train)
    # Visualize the learned perceptron weights.
    plot_visualization(perceptron.w[:-1, :], class_names, None)
    # Classify the test sets.
    accuracy, y_pred = perceptron.test(x_test, y_test)
    print('Perceptron avg_acc: ', accuracy)
    # Plot confusion matrix.
    plot_confusion_matrix(y_test, y_pred, classes=class_names,
                          normalize=True, title='Confusion matrix, with normalization')
    plt.show()

    # ------------------add: plot highest/lowest posterior probabilities
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    diagnal_li = [cm[i, i] for i in range(num_class)]
    highest_class, lowest_class = np.argmax(diagnal_li), np.argmin(diagnal_li)
    tmp_arr1, tmp_arr2 = y_test == highest_class, y_test == lowest_class
    sample_idx1 = np.random.choice(np.where(tmp_arr1 == True)[0], 5)
    sample_idx2 = np.random.choice(np.where(tmp_arr2 == True)[0], 5)
    sample_idx = np.hstack([sample_idx1, sample_idx2])
    sample_x, sample_y = x_test[sample_idx], y_test[sample_idx]
    plot_samples(sample_x, sample_y, class_names, "Greys")
    # ------------------add: plot highest/lowest posterior probabilities
    '''
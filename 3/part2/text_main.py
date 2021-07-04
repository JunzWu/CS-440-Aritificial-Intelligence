# text_main.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

import csv
from TextClassifier import TextClassifier, list_divide
import string
import plot

"""
This file contains the main application that is run for this part of the MP.
No need to modify this file
"""


def read_stop_words(filename):
    """
       Reads in the stop words which are used for data preprocessing
       Returns a set of stop words
    """
    stop_words = set()
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            for word in row:
                stop_words.add(word.strip(" '"))
    stop_words.remove('')

    return stop_words


def readFile(filename, stop_words):
    """
    Loads the files in the folder and returns a list of lists of words from
    the text in each file and the corresponding labels
    """
    translator = str.maketrans("", "", string.punctuation)
    with open(filename) as csv_file:
        labels = []
        data = []
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            labels.append(int(row[0]))
            row[2] = row[2].lower()
            text = row[2].translate(translator).split()
            text = [w for w in text if w not in stop_words]
            data.append(text)

    return data, labels


def load_dataset(data_dir=''):
    """

    :param data_dir: directory path to your data
    :return: both the train and test data sets
    """
    stop_words = read_stop_words(data_dir + 'stop_words.csv')
    x_train, y_train = readFile(data_dir + 'train_text.csv', stop_words)
    x_test, y_test = readFile(data_dir + 'dev_text.csv', stop_words)

    return x_train, y_train, x_test, y_test


def compute_results(actual_labels, pred_labels, show_class_names=False, show_conf_matrix=False):
    """

    :param actual_labels: Gold labels for the given texts
    :param pred_labels: Predicted Labels for the given texts
    """
    precision = []
    recall = []
    for c in range(1, 15):
        actual_c = {i for i in range(
            len(actual_labels)) if actual_labels[i] == c}
        pred_c = {i for i in range(len(pred_labels)) if pred_labels[i] == c}
        tp = len(actual_c.intersection(pred_c))

        if len(pred_c) > 0:
            precision.append(tp / len(pred_c))
        else:
            precision.append(0.0)

        recall.append(tp / len(actual_c))

    f1 = [2 * (p * r) / (p + r) if (p + r) !=
          0.0 else 0.0 for p, r in zip(precision, recall)]

    # if show class name?
    if not show_class_names:
        print ("Precision for all classes :", precision)
        print ("Recall for all classes:", recall)
        print ("F1 Score for all classes:", f1)
    else:
        class_names = read_class_names()
        print ("---- Precision for all classes ----")
        for name, val in zip(class_names, precision):
            print('%s: %.3f' % (name, val))
        print ("---- Recall for all classes ----")
        for name, val in zip(class_names, recall):
            print('%s: %.3f' % (name, val))
        print ("---- F1 Score for all classes ----")
        for name, val in zip(class_names, f1):
            print('%s: %.3f' % (name, val))

    # if show confusion matrix?
    if show_conf_matrix:
        # initialize cm
        conf_matrix_cnt = [[0] * 14 for _ in range(14)]
        conf_matrix = [[0] * 14 for _ in range(14)]
        for t in range(1, 15):
            actual_c_idxs = {i for i in range(
                len(actual_labels)) if actual_labels[i] == t}
            for idx in actual_c_idxs:
                p = pred_labels[idx]
                conf_matrix_cnt[t - 1][p - 1] += 1

        for i, row in enumerate(conf_matrix_cnt):
            conf_matrix[i] = list_divide(row, sum(row), precision=3)
        print('-------- confusion matrix --------')
        plot.display_2d_list(conf_matrix)
        # print(conf_matrix)


def read_class_names():
    class_names = []
    with open('classes.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_names.append(line)
    return class_names


if __name__ == '__main__':
    # import ipdb
    # ipdb.set_trace()
    x_train, y_train, x_test, y_test = load_dataset()

    # 1.Unigram Model, lambda_mix = 0, determine which laplace is best
    laplace_k_l = [round(0.1 * i, 2) for i in range(1, 21)]
    avg_acc_l = []
    for i, laplace_k in enumerate(laplace_k_l):
        MNB = TextClassifier(laplace_k=laplace_k)
        MNB.fit(x_train, y_train)
        accuracy, _ = MNB.predict(x_test, y_test, lambda_mix=0)
        avg_acc_l.append(accuracy)
        print('\rexperiment with different laplace_k...... (%d/%d)' %
              (i + 1, len(laplace_k_l)), end='')
    plot.plot_single_line(laplace_k_l, avg_acc_l, xlabel='laplace_k',
                          ylabel='average_acc', title='average_acc ~ laplace_k', xticks=laplace_k_l)
    best_acc_idx = avg_acc_l.index(max(avg_acc_l))
    print('\nBest laplace_k is %f, best acc is %f' %
          (laplace_k_l[best_acc_idx], max(avg_acc_l)))

    # 2.Unigram Model result
    # MNB = TextClassifier(laplace_k=0.2)
    # MNB.fit(x_train, y_train)
    # accuracy, pred = MNB.predict(x_test, y_test)
    # compute_results(y_test, pred, show_class_names=True, show_conf_matrix=True)
    # print("Accuracy {0:.4f}".format(accuracy))
    # print('-------- top feature words --------')
    # plot.display_2d_list(MNB.top_features(), ele_length=12)

    # 3. The change in accuracy results when the class prior changes
    # MNB = TextClassifier(laplace_k=0.2)
    # MNB.fit(x_train, y_train)
    # for prior_type in ['normal', 'random', 'none']:
    #     accuracy, _ = MNB.predict(x_test, y_test, prior_type=prior_type)
    #     print('type: %s, acc: %.3f' % (prior_type, accuracy))

    # 4. Extra Credit  mix with bi-gram
    # MNB = TextClassifier(laplace_k=0.2)
    # MNB.fit(x_train, y_train)
    # lambda_mix_l = [round(0.01 * i, 2) for i in range(0, 101)]
    # avg_acc_l = []
    # for i, lambda_mix in enumerate(lambda_mix_l):
    #     accuracy, _ = MNB.predict(x_test, y_test, lambda_mix=lambda_mix)
    #     avg_acc_l.append(accuracy)
    #     print('\rexperiment with different lambda_mix...... (%d/%d)' %
    #           (i + 1, len(lambda_mix_l)), end='')
    # plot.plot_single_line(lambda_mix_l, avg_acc_l, xlabel='lambda_mix',
    #                       ylabel='average_acc', title='average_acc ~ lambda_mix')
    # best_acc_idx = avg_acc_l.index(max(avg_acc_l))
    # print('\nBest lambda_mix is %.2f, best acc is %f' %
    #       (lambda_mix_l[best_acc_idx], max(avg_acc_l)))

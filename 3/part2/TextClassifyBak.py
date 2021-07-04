# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from math import log
from tqdm import tqdm
import numpy as np


def split_by_ngram(doc, ngram):
    result = []
    for i in range(len(doc) - ngram + 1):
        result.append(tuple(doc[i + j] for j in range(ngram)))
    return result


def list_add(li1, int_or_li, replace=False):
    '''simulate element-wise add in numpy'''
    res = []
    if isinstance(int_or_li, int):
        for i, _ in enumerate(li1):
            res.append(li1[i] + int_or_li)
    elif isinstance(int_or_li, list):
        assert len(li1) == len(int_or_li)
        for i, v in enumerate(int_or_li):
            res.append(li1[i] + v)

    if replace == True:  # 替代模式
        li1 = res
    else:
        return res


def get_vocabulary_dic(docs, n_gram):
    vocab_set = set()
    for doc in docs:
        doc = split_by_ngram(doc, n_gram)
        vocab_set = vocab_set | set(doc)
    vocab_dic = {v: i for i, v in enumerate(vocab_set)}
    return vocab_dic


class TextClassifier(object):
    def __init__(self, train_set):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

        # make vocabulary dict, save dict length
        self.prior = None

    def doc_to_wordVec(self, doc):
        '''transform one vocabulary doc into wordVec type'''
        result = [0] * self.feature_num
        # result = [1] * self.feature_num
        for vocab in doc:
            if vocab in self.vocab_dic:
                result[self.vocab_dic[vocab]] += 1
        return result

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here
        # 1.make vocabulary dict, save parameter
        self.vocab_dic = get_vocabulary_dic(train_set)
        self.feature_num = len(self.vocab_dic)
        self.class_num = len(set(train_label))
        self.likelihood = [[1 for _ in range(self.feature_num)]
                           for _ in range(self.class_num)]
        # self.class_sum = list(
        #     map(lambda l: sum(l), self.likelihood))  # laplace smooth
        self.class_sum = [2] * self.class_num

        # 2.trasform train_set and train_label
        train_set = list(map(lambda doc: self.doc_to_wordVec(doc), train_set))
        # label should be 0-index style
        train_label = list(map(lambda l: l - 1, train_label))

        # 3.get prior probablity
        self.prior = [0] * self.class_num
        for label in train_label:
            self.prior[label] += 1

        self.prior = list(map(lambda x: x / len(train_label), self.prior))

        # 4.get conditional probability - likelihood
        for docVec, docLabel in tqdm(zip(train_set, train_label), total=len(train_set), desc='training....'):
            self.likelihood[docLabel] = list_add(
                self.likelihood[docLabel], docVec)
            self.class_sum[docLabel] += sum(docVec)

        for i in range(self.class_num):
            for j in range(self.feature_num):
                self.likelihood[i][j] = log(
                    self.likelihood[i][j] / self.class_sum[i])
                # self.likelihood[i][j] /= self.class_sum[i]

    def argmax(self, li):
        '''get the max ele index in list arg li'''
        max_i, max_v = 0, li[0]
        for i, v in enumerate(li):
            if v > max_v:
                max_i = i
        return max_i

    def predict(self, dev_set, dev_label, lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """
        # TODO: Write your code here

        # trasform dev_set and dev_label
        dev_set = list(map(lambda doc: self.doc_to_wordVec(doc), dev_set))
        # label should be 0-index style
        # dev_label = list(map(lambda l: l - 1, dev_label))
        predict_label = []
        for doc_vec in tqdm(dev_set, total=len(dev_set), desc='testing....'):
            log_probs = list(map(lambda x: log(x), self.prior))
            for y_i in range(self.class_num):
                for f_i in range(self.feature_num):
                    # log_probs[y_i] += log(self.likelihood[y_i]
                    #                       [f_i] * doc_vec[f_i])
                    log_probs[y_i] += self.likelihood[y_i][f_i] * doc_vec[f_i]
            this_predict_label = self.argmax(log_probs)
            predict_label.append(this_predict_label + 1)

        accuracy, cnt = 0.0, 0
        assert len(predict_label) == len(dev_label)
        for t, p in zip(dev_label, predict_label):
            if t == p:
                cnt += 1
        accuracy = cnt / len(predict_label)

        return accuracy, predict_label

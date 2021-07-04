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
# from math import log
import math
from collections import Counter, defaultdict
import random


def split_by_ngram(doc, ngram):
    doc = ['<s>'] + doc
    result = []
    for i in range(len(doc) - ngram + 1):
        result.append(tuple(doc[i + j] for j in range(ngram)))
    return result


def list_add(li1, int_or_li, replace=False):
    '''simulate element-wise add in numpy'''
    res = []
    if isinstance(int_or_li, (int, float)):
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


def list_multiply(li1, int_or_li, replace=False):
    '''simulate element-wise add in numpy'''
    res = []
    if isinstance(int_or_li, (int, float)):
        for i, _ in enumerate(li1):
            res.append(li1[i] * int_or_li)
    elif isinstance(int_or_li, list):
        assert len(li1) == len(int_or_li)
        for i, v in enumerate(int_or_li):
            res.append(li1[i] * v)

    if replace == True:  # 替代模式
        li1 = res
    else:
        return res


def list_divide(li1, int_or_li, replace=False, precision=5):
    '''simulate element-wise add in numpy'''
    res = []
    if isinstance(int_or_li, (int, float)):
        for i, _ in enumerate(li1):
            res.append(round(li1[i] / int_or_li, precision))
    elif isinstance(int_or_li, list):
        assert len(li1) == len(int_or_li)
        for i, v in enumerate(int_or_li):
            res.append(round(li1[i] / v, precision))

    if replace == True:  # 替代模式
        li1 = res
    else:
        return res


def list_log(li):
    '''apply log to list element'''
    return list(map(lambda x: math.log(x), li))


def argmax(li):
    '''get the max ele index in list arg li'''
    max_i, max_v = 0, li[0]
    for i, v in enumerate(li):
        if v > max_v:
            max_i = i
    return max_i


class TextClassifier(object):
    def __init__(self, laplace_k=0.05):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0
        self.n_gram = 2
        self.laplace_k = laplace_k

    def doc_to_wordVec(self, doc):
        '''transform one vocabulary doc into wordVec type'''
        result = [0] * self.feature_num
        # result = [1] * self.feature_num
        for vocab in doc:
            if vocab in self.vocab_dic:
                result[self.vocab_dic[vocab]] += 1
        return result

    def corpus_stat(self, train_set, train_label):
        '''
        statstics funtion of the corpus
        vocab_cnt: stat sigle vocab count for 10 class
        gram_cnt:stat n-gram vocab count
        class_cnt: stat all num in each class
        '''
        vocab_cnt = defaultdict(lambda: [self.laplace_k] * self.class_num)
        vocab_prob = defaultdict(lambda: [0] * self.class_num)
        gram_cnt = defaultdict(lambda: [self.laplace_k] * self.class_num)
        gram_prob = defaultdict(lambda: [0] * self.class_num)
        class_cnt = [self.laplace_k *
                     len(vocab_cnt)] * self.class_num  # laplace smooth
        # training --- stat four dict
        for doc, label in zip(train_set, train_label):
            for v, c in Counter(doc).items():
                vocab_cnt[v][label] += c
                class_cnt[label] += c
            gram_doc = split_by_ngram(doc, self.n_gram)
            for v, c in Counter(gram_doc).items():
                gram_cnt[v][label] += c

        for k, v_l in vocab_cnt.items():
            vocab_prob[k] = list_divide(v_l, class_cnt)
        for k, v_l in gram_cnt.items():
            pre_vl = vocab_cnt[k[0]]
            if '<s>' in k:
                gram_prob[k] = vocab_prob[k[1]]  # P(w1) for fist item
            else:
                # P(wn|wn−1) for next item
                gram_prob[k] = list_divide(v_l, pre_vl)

        self.vocab_cnt, self.vocab_prob, self.gram_cnt, self.gram_prob = vocab_cnt, vocab_prob, gram_cnt, gram_prob

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

        # 1.get prior probablity
        self.class_num = len(set(train_label))
        train_label = list(map(lambda l: l - 1, train_label)
                           )  # label should be 0-index style
        self.prior = [0] * self.class_num
        for label in train_label:
            self.prior[label] += 1
        self.prior = list(map(lambda x: x / len(train_label), self.prior))

        # 2.calculate count and probablity
        self.corpus_stat(train_set, train_label)

    def top_features(self, top_k=20):
        vocab_counter = [Counter() for _ in range(self.class_num)]
        for vocab, vocab_l in self.vocab_cnt.items():
            for i, num in enumerate(vocab_l):
                vocab_counter[i].update({vocab: num})
        top_k_features = list(
            map(lambda x: x.most_common(top_k), vocab_counter))
        top_k_features = list(
            map(lambda li: list(map(lambda x: x[0], li)), top_k_features))
        return top_k_features

    def predict(self, dev_set, dev_label, lambda_mix=0.0, prior_type='normal'):
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
        predict_label = []
        # testing....:
        for doc in dev_set:
            # set log_probs as prior_type
            if prior_type == 'normal':
                log_probs = list_log(self.prior)
            elif prior_type == 'random':
                log_probs = list_log([random.random()
                                      for _ in range(self.class_num)])
            elif prior_type == 'none':
                log_probs = [0] * self.class_num
            # Unigram part
            uni_vlog = [0] * self.class_num
            for v, c in Counter(doc).items():
                if v in self.vocab_prob:
                    this_prob = list_multiply(self.vocab_prob[v], c)
                    this_vlog = list_log(this_prob)
                    uni_vlog = list_add(uni_vlog, this_vlog)
            uni_vlog = list_multiply(uni_vlog, 1 - lambda_mix)
            # Bigram part
            bi_vlog = [0] * self.class_num
            gram_doc = split_by_ngram(doc, self.n_gram)
            for bv, c in Counter(gram_doc).items():
                if bv in self.gram_prob:

                    this_prob = list_multiply(self.gram_prob[bv], c)
                    this_vlog = list_log(this_prob)
                    bi_vlog = list_add(bi_vlog, this_vlog)
            bi_vlog = list_multiply(bi_vlog, lambda_mix)

            # log_probs = list_multiply(list_add(uni_vlog, bi_vlog), self.prior)
            log_probs = list_add(list_add(uni_vlog, bi_vlog), log_probs)
            this_predict_label = argmax(log_probs)
            predict_label.append(this_predict_label + 1)

        accuracy, cnt = 0.0, 0
        assert len(predict_label) == len(dev_label)
        for t, p in zip(dev_label, predict_label):
            if t == p:
                cnt += 1
        accuracy = cnt / len(predict_label)

        return accuracy, predict_label

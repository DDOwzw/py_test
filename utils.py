#!/usr/local/bin/python3
import random
import sklearn.metrics.pairwise as pairwise
import cvxopt as co
import numpy as np

def generateSamples(X, sample_num, sample_size):
    sampleSets = [[] for _ in range(sample_num)]
    indexSets = [[] for _ in range(sample_num)]
    for i in range(sample_num):
        for j in range(sample_size):
            idx = random.randint(0, len(X) - 1)
            sampleSets[i].append(X[idx])
            indexSets[i].append(idx)
    return sampleSets, indexSets

def partition(X, part_num, part_size):
    dataSet = []
    if part_num:
        part_size = int(np.ceil(len(X) / part_num))
        for i in range(0, len(X), part_size):
            if i + part_size <= len(X):
                dataSet.append(X[i:i + part_size])
            else:
                dataSet.append([])
                for _ in range(part_size):
                    dataSet[-1].append(X[random.randint(i, len(X)-1)])
    else:
        cur = 0
        while cur < len(X):
            if cur + part_size <= len(X):
                dataSet.append(X[cur:cur + part_size])
            else:
                dataSet.append([])
                for _ in range(part_size):
                    dataSet[-1].append(X[random.randint(cur, len(X)-1)])

            cur += part_size

    return dataSet

def calcSigma(X_train):
    dist = []
    for i in range(len(X_train)):
         for j in range(i + 1, len(X_train)):
             diff = np.array(X_train[i]) - np.array(X_train[j])
             dist.append(np.sqrt(sum(diff ** 2)))
    return np.median(dist)

def KMM(X_train, X_test, sigma):
    n_tr = len(X_train)
    n_te = len(X_test)

    # n_tr x n_tr
    X_tr_tr = pairwise.rbf_kernel(X_train, X_train, sigma)
    X_tr_tr = 0.5 * (X_tr_tr + X_tr_tr.transpose())

    # n_tr x n_te -> n_tr x 1
    X_tr_te = pairwise.rbf_kernel(X_train, X_test, sigma)
    ones = np.ones(shape = (n_te, 1))
    X_tr_te = -(n_tr / n_te) * np.dot(X_tr_te, ones)

    epsilon = (np.sqrt(n_tr) - 1) / np.sqrt(n_tr)

    _1 = np.ones(shape=(1, n_tr))
    __1 = -np.ones(shape=(1, n_tr))
    # (n_tr * 2 + 2) x n_tr
    A = np.vstack([_1, __1, -np.eye(n_tr), np.eye(n_tr)])
    B = np.array([[n_tr * (epsilon+1), n_tr*(epsilon-1)]])
    # (2 + n_tr * 2) x 1
    B = np.vstack([B.T, -np.zeros(shape=(n_tr, 1)), np.ones(shape=(n_tr,1))*1000])

    mX_tr_tr = co.matrix(X_tr_tr, tc='d')
    mX_tr_te = co.matrix(X_tr_te, tc='d')
    mA = co.matrix(A, tc='d')
    mB = co.matrix(B, tc='d')
    co.solvers.options['show_progress'] = False
    beta = co.solvers.qp(mX_tr_tr, mX_tr_te, mA, mB)

    return [i for i in beta['x']]

# written by lsmgeb89
def eval(beta, true_beta):
  if (beta.size is 0) or (true_beta.size is 0) or (beta.size != true_beta.size):
    raise RuntimeError("Empty or different size array")

  if (beta.dtype is not np.dtype('float')) or (true_beta.dtype is not np.dtype('float')):
    raise RuntimeError("Array type is not float")

  sum_beta = np.sum(beta)
  sum_true_beta = np.sum(true_beta)

  nor_beta = beta / sum_beta
  nor_true_beta = true_beta / sum_true_beta

  diff = nor_beta - nor_true_beta
  sq_diff = np.power(diff, 2)

  return np.sum(sq_diff) / sq_diff.size

def readData(filename):
    import pickle
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data

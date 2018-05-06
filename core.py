#!/usr/local/bin/python3
import numpy as np
from utils import *

def cenKMM(X_train, X_test, sigma):
    return KMM(X_train, X_test, sigma)

def ensKMM(X_train, X_test, k, sigma):
    retBeta = np.zeros((len(X_train)))
    X_test_parts = partition(X_test, k, None)
    for part in X_test_parts:
        beta = np.array(KMM(X_train, part, sigma))
        retBeta += len(part) / len(X_test) * beta
    return retBeta

def SKMM(X_train, X_test, k, eta, sigma):
    m = len(X_train) // k
    s = int(np.ceil(np.log(eta) / (m * np.log(1 - 1 / len(X_train)))))
    sample_sets, index_sets = generateSamples(X_train, s, m)
    test_parts = partition(X_test, k, None)

    idx2beta = {}
    for i in range(len(X_train)):
        idx2beta[i] = [0, 0] #sum, count

    for i in range(s):
        for j in range(k):
            beta = KMM(sample_sets[i], test_parts[j], sigma)
            for a in range(len(beta)):
                idx2beta[index_sets[i][a]][0] += beta[a]
                idx2beta[index_sets[i][a]][1] += 1

    retBeta = np.zeros((len(X_train)))
    for i in idx2beta.keys():
        if idx2beta[i][1] != 0:
            retBeta[i] = idx2beta[i][0] / idx2beta[i][1]

    return retBeta

def SDKMM_helper(train, test, indices, sigma):
    beta = KMM(train, test, sigma)
    ret = []
    for i in range(len(indices)):
        ret.append([indices[i], beta[i]])
    return ret

def SDKMM(sc, X_train, X_test, k, eta, sigma):
    m = len(X_train) // k
    s = int(np.ceil(np.log(eta) / (m * np.log(1 - 1 / len(X_train)))))
    sample_sets, index_sets = generateSamples(X_train, s, m)
    test_parts = partition(X_test, k, None)

    pair_sets = []
    for i in range(len(sample_sets)):
        pair_sets += [[i, j] for j in range(len(test_parts))]

    sc.addPyFile('src/core.py')
    sc.addPyFile('src/utils.py')
    train_sets = sc.broadcast(sample_sets)
    train_indices = sc.broadcast(index_sets)
    test_sets = sc.broadcast(test_parts)
    sigma_bc = sc.broadcast(sigma)

    rdd = sc.parallelize(pair_sets)
    beta_pair = rdd.flatMap(lambda x: SDKMM_helper(train_sets.value[x[0]], test_sets.value[x[1]], train_indices.value[x[0]], sigma_bc.value))\
                .groupByKey()\
                .mapValues(lambda x: sum(list(x)) / len(x))\
                .collect()
    beta = np.zeros(len(X_train))
    for i in range(len(beta_pair)):
        beta[beta_pair[i][0]] = beta_pair[i][1]
    return beta
    # idx2beta = {}
    # for i in range(len(X_train)):
    #     idx2beta[i] = [0, 0] #sum, count
    #
    # for i in range(s):
    #     for j in range(k):
    #         beta = KMM(sample_sets[i], test_parts[j], sigma)
    #         for a in range(len(beta)):
    #             idx2beta[index_sets[i][a]][0] += beta[a]
    #             idx2beta[index_sets[i][a]][1] += 1
    #
    # retBeta = np.zeros((len(X_train)))
    # for i in idx2beta.keys():
    #     if idx2beta[i][1] != 0:
    #         retBeta[i] = idx2beta[i][0] / idx2beta[i][1]
    #
    # return retBeta

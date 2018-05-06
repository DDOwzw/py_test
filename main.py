#!/usr/local/bin/python3
from utils import *
from core import *
import numpy as np
from pyspark.context import SparkContext
import matplotlib.pyplot as plt

def readDataSets(size):
    train = readData('dataset/powersupply/' + str(size) + '_train.txt')
    test = readData('dataset/powersupply/' + str(size) + '_test.txt')
    true_beta = np.array(readData('dataset/powersupply/' + str(size) + '_beta.txt'))
    return train, test, true_beta

# Need the train data and test data as input
def computeTrueBeta(train, test):
    sigma = compute_sigma(train)
    res = kmm(train,test, sigma)

sc = SparkContext()
sizes = [100, 300, 500, 1000, 1500, 2000, 2500, 3000]
cenRes, ensRes, sdkmmRes = [], [], []
for size in sizes:
    print(size)
    train, test, true_beta = readDataSets(size)
    sigma = calcSigma(train)
    beta = np.array(cenKMM(train, test, sigma))
    true_beta = beta
    beta2 = np.array(ensKMM(train, test, 10, sigma))
    beta3 = np.array(SDKMM(sc, train, test, 10, 0.01, sigma))
    cenRes.append(np.log(eval(beta, true_beta)))
    ensRes.append(np.log(eval(beta2, true_beta)))
    sdkmmRes.append(np.log(eval(beta3, true_beta)))
print(cenRes)
print(ensRes)
print(sdkmmRes)

cenline = plt.plot(sizes, cenRes, label = 'CenKMM', color='red', marker = '^', linestyle = '--')[0]
ensline = plt.plot(sizes, ensRes, label = 'EnsKMM', color='black', marker = 's', linestyle = '--')[0]
sdkmmline = plt.plot(sizes, sdkmmRes, label = 'SKMM', color='blue', marker = '.', linestyle = '-')[0]
plt.legend(handles = [cenline, ensline, skmmline])

# first off we load up some modules we want to use
import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
import collections
import theautil

# setup logging
logging.basicConfig(stream = sys.stderr, level=logging.INFO)

mupdates = 1000
data = np.loadtxt("iris.csv", delimiter=",")
inputs  = data[0:,0:4].astype(np.float32)
outputs = data[0:,4:5].astype(np.int32)
tsize = len(outputs)
theautil.joint_shuffle(inputs,outputs)

train_and_valid, test = theautil.split_validation(90, inputs, outputs)
train, valid = theautil.split_validation(90, train_and_valid[0], train_and_valid[1])

def linit(x):
    return x.reshape((len(x),))

train = (train[0],linit(train[1]))
valid = (valid[0],linit(valid[1]))
test  = (test[0] ,linit(test[1]))

# build our classifier
print "We're building a RBM of 1 input layer node, 4 hidden layer nodes, and an output layer of 4 nodes. The output layer has 4 nodes because we have 4 classes that the neural network will output."

cnet = theanets.Classifier([4,10,5,2])
cnet.train(train,valid, algo='layerwise', patience=1, max_updates=mupdates)
cnet.train(train,valid, algo='rprop', patience=1, max_updates=mupdates)

print "Learner on the test set"
classify = cnet.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])

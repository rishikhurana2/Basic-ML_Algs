"""
Implementation of k-nearest-neighbor classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from binary import *


class KNN(BinaryClassifier):
    """
    This class defines a nearest neighbor classifier, that support
    _both_ K-nearest neighbors _and_ epsilon ball neighbors.
    """

    def __init__(self, opts):
        """
        Initialize the classifier.  There's actually basically nothing
        to do here since nearest neighbors do not really train.
        """

        # remember the options
        self.opts = opts

        # just call reset
        self.reset()

    def reset(self):
        self.trX = np.zeros((0,0))    # where we will store the training examples
        self.trY = np.zeros((0))      # where we will store the training labels

    def online(self):
        """
        We're not online
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)
    
    def distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x2[i] - x1[i])**2
        return distance

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the 'vote' in favor of a positive
        or negative label.  In particular, if, in our neighbor set,
        there are 5 positive training examples and 2 negative
        examples, we return 5-2=3.

        Everything should be in terms of _Euclidean distance_, NOT
        squared Euclidean distance or anything more exotic.
        """

        isKNN = self.opts['isKNN']     # true for KNN, false for epsilon balls
        N     = self.trX.shape[0]      # number of training examples

        if self.trY.size == 0:
            return 0                   # if we haven't trained yet, return 0
        elif isKNN:
            # this is a K nearest neighbor model
            # hint: look at the 'argsort' function in numpy
            K = self.opts['K']         # how many NN to use

            val = 0                    # this is our return value: #pos - #neg of the K nearest neighbors of X
            distances = [] # array of tuples that is (distance, point)
            for i in range(len(self.trX)):
                distances.append((self.distance(X, self.trX[i]), self.trY[i]))
            distances.sort(key = lambda x : x[0]) # sort array by distance
            KNN = distances[0 : K]                # K nearest neighbrs is here
            Y_KNN = np.array([knn[1] for knn in KNN])
            val = sum(Y_KNN == 1) - sum(Y_KNN == -1)
            
            return val
        else:
            # this is an epsilon ball model
            eps = self.opts['eps']     # how big is our epsilon ball        
            val = 0                    # this is our return value: #pos - #neg within and epsilon ball of X
            for i in range(len(self.trX)):
                if self.distance(X, self.trX[i]) <= eps**2:
                    val += self.trY[i]

            ### TODO: YOUR CODE HERE
            return val
                
            


    def getRepresentation(self):
        """
        Return the weights
        """
        return (self.trX, self.trY)

    def train(self, X, Y):
        """
        Just store the data.
        """
        self.trX = X
        self.trY = Y
        

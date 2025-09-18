"""
In dt.py, you will implement a basic decision tree classifier for
binary classification.  Your implementation should be based on the
minimum classification error heuristic (even though this isn't ideal,
it's easier to code than the information-based metrics).
"""

import numpy as np

from binary import *
import util
from collections import Counter

class DT(BinaryClassifier):
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)
        
        self.isLeaf = True
        self.label  = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)
    
    # Given data sets X and Y, find the best binary feature to split from.
        # i.e. the binary feature such that its split allows us to get the most info gain
    # Return the splitted data sets
    def getBestFeature(self, X, Y, used_features):
        num_features = X.shape[1]
        max_score = 0

        best_feature_index = -1
        best_X_zero_subset = None
        best_X_one_subset  = None

        best_Y_zero_subset = None
        best_Y_one_subset  = None

        for i in range(num_features):
            X_zero = []
            Y_zero = np.array([])

            X_one = []
            Y_one = np.array([])

            for j in range(X.shape[0]):
                # if the ith feature is zero, put it in X_zero, Y_zero
                if X[j][i] == 0:
                    X_zero.append(X[j])
                    Y_zero = np.append(Y_zero, Y[j])
                else: # other wise, put it in X_one, Y_one
                    X_one.append(X[j])
                    Y_one = np.append(Y_one, Y[j])
            # compute the score bia splitting on the current feature
            element_count_zero = Counter(Y_zero)
            element_count_one  = Counter(Y_one)            
            if element_count_zero.most_common(1):
                most_frequent_Y_zero = element_count_zero.most_common(1)[0][1]
            else:
                most_frequent_Y_zero = 0              
            if element_count_one.most_common(1):
                most_frequent_Y_one  = element_count_one.most_common(1)[0][1]
            else:
                most_frequent_Y_one = 0
            # curr score is how confident we can predict the output
            curr_score = float(float(most_frequent_Y_zero) + float(most_frequent_Y_one))/float(len(Y_one) + len(Y_zero))
            # if the score is better, then record the feature
            if curr_score > max_score and i not in used_features:
                max_score = curr_score
                best_feature_index = i
                
                best_X_zero_subset = np.array(X_zero)
                best_X_one_subset  = np.array(X_one)

                best_Y_zero_subset = Y_zero
                best_Y_one_subset  = Y_one

        return best_feature_index, best_X_zero_subset, best_Y_zero_subset, best_X_one_subset, best_Y_one_subset            
        

    def trainDT(self, X, Y, maxDepth, used):
        """
        recursively build the decision tree
        """

        # get the size of the data set

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf = True    ### TODO: YOUR CODE HERE

            self.label  = Y[0]    ### TODO: YOUR CODE HERE


        else:
            # get the best feature for current data set
            
            # we need to find a feature to split on
            bestFeature, best_X_left, best_Y_left, best_X_right, best_Y_right = self.getBestFeature(X, Y, used)
            used.add(bestFeature)


            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  = False    ### TODO: YOUR CODE HERE
                self.feature = bestFeature    ### TODO: YOUR CODE HERE


                self.left  = DT({'maxDepth': maxDepth-1})
                self.right = DT({'maxDepth': maxDepth-1})
                # recurse on our children by calling
                #   self.left.trainDT(...) 
                # and
                #   self.right.trainDT(...) 
                # with appropriate arguments
                ### TODO: YOUR CODE HERE
                self.left.trainDT(best_X_left, best_Y_left, maxDepth - 1, used)
                self.right.trainDT(best_X_right, best_Y_right, maxDepth - 1, used)        
    def predict(self, X):
        """
        Traverse the tree to make predictions.  You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch.
        """
        if self.isLeaf:
            return self.label
        if X[self.feature] == 1:
            return self.right.predict(X)
        else:
            return self.left.predict(X)
            

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['maxDepth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponting classes,
            say Y(X(:,5)>=0.5) and if you want the correspnding rows of X,
            say X(X(:,5)>=0.5,:)
            
          - i suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - take a look at the 'mode' and 'uniq' functions in util.py
        """
        
        self.trainDT(X, Y, self.opts['maxDepth'], set())


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        
        return self


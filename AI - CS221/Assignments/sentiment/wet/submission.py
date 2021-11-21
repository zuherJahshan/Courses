#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    x_list = x.split()
    phi = dict()
    for word in x_list:
        if word in phi.keys():
            phi[word] += 1
        else:
            phi[word] = 1
    return phi
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def getVal(feature, in_dict):
    if feature in in_dict.keys():
        return in_dict[feature]
    else:
        return 0


def getScore(phi, weights):
    score = 0
    for feature in phi.keys():
        score += getVal(feature, phi) * getVal(feature, weights)
    return score


def updateWeights(phi, weights, y, eta):
    for feature in phi.keys():
        if feature in weights.keys():
            weights[feature] += eta*y*phi[feature]
        else:
            weights[feature] = eta*y*phi[feature]
    return weights

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    for i in range(numIters):
        for example in trainExamples:
            x = example[0] # string
            y = example[1] # real outcome
            phi = extractWordFeatures(x) # feature vector
            score = getScore(phi, weights) # score of the string x
            hinge_loss = 1 - score*y # the hinge loss of x
            if hinge_loss > 0: # if it is smaller than zero, we are great
                updateWeights(phi, weights, y, eta)
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        phi = {}
        for feature in weights.keys():
            if random.random() < 0.5:
                phi[feature] = random.randint(1, 10)
        score = getScore(phi, weights)
        if score > 0:
            y = 1
        elif score < 0:
            y = -1
        else:
            return generateExample()
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        phi = {}
        fixed_x = ''.join(x.split())
        for i in range(len(x) - n):
            if fixed_x[i:(i+n)] in phi:
                phi[fixed_x[i:(i+n)]] += 1
            else:
                phi[fixed_x[i:(i + n)]] = 1
        return phi
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################

def getDistance(vec1, vec2):
    dist = 0
    united_vec = vec1 | vec2
    for feature in united_vec.keys():
        dist += pow(getVal(feature, vec1) - getVal(feature, vec2), 2)
    return dist


def addFeatures(phi1, phi2):
    united_phi = phi1 | phi2
    for feature in united_phi.keys():
        united_phi[feature] = getVal(feature, phi1) + getVal(feature, phi2)
    return united_phi

def devFeatures(phi, d):
    for feature in phi.keys():
        phi[feature] = phi[feature] / d
    return phi

def updateAssignments(examples, assignments, centroids):
    loss = 0
    for example_idx in range(len(examples)):
        min_dist = -1
        for center_idx in range(len(centroids)):
            curr_dist = getDistance(centroids[center_idx], examples[example_idx])
            if curr_dist < min_dist or min_dist == -1:
                min_dist = curr_dist
                assignments[example_idx] = center_idx
        loss += min_dist
    return [loss, assignments]

def updateCentroids(examples, assignments, centroids):
    new_centroids = [{}]*len(centroids)
    assignment_count = [0]*len(centroids)
    for i in range(len(assignments)):
        new_centroids[assignments[i]] = addFeatures(new_centroids[assignments[i]], examples[i])
        assignment_count[assignments[i]] += 1
    for i in range(len(new_centroids)):
        if assignment_count[i] == 0:
            new_centroids[i] = centroids[i]
        new_centroids[i] = devFeatures(new_centroids[i], assignment_count[i])
    return new_centroids


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    '''
    examples = dict of feature, val
    , K is the number of clusters available
    maxIters, is the maximum number of iterations
    '''
    centroids = random.choices(examples, k=K)
    assignments = list([-1] * len(examples))
    prev_loss = loss = -1
    for i in range(maxIters):
        loss, assignments = updateAssignments(examples, assignments, centroids)
        centroids = updateCentroids(examples, assignments, centroids)
        if loss == prev_loss:
            break
        prev_loss = loss
    return [centroids, assignments, loss]
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

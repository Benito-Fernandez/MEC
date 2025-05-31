#
# pyDefinitions.py
# NeuralNetworkStandardLibrary
#
# Created by Benito Fernandez, Alejandro Fernandez, Juan Rincon on 5/09/18.
# Other Authors: Luis Fernandez 6/14/18
# Copyright (?)
#

'''
This file is comprised of 5 main sections: Activation functions, input operators, loss functions, Axon Input Operators, and Weight Type Initializations

Dictionaries are used to access these methds. The following key-value pair convention should be used for every dictionary:
    - key: activationName as String
    - value: (activationFunctionName, activationFunctionName_, activationName, activationName_)
        - value types: (method, method, string, string)

Some sources:
-   http://www.rueckstiess.net/research/snippets/show/72d2363e
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""=------------------------------------------------------------------------"""
__path__ = ["./"]
global pyNetDefinitionsLoaded,  pyNetConstantsLoaded,  pyNetUtilitiesLoaded,\
       pyNetFunctionsLoaded,    pyNetPandasDataLoaded, pyNetNeuralNetworksLoaded,\
       pyNetFilesDictLoaded,    pyNetDBLoaded
global xDebug
xDebug = False
pyNetDefinitionsLoaded = True

if xDebug: print("[pyDefinitions]> loading pyDefinitions.py module ...")

"""=------------------------------------------------------------------------"""
# ----------------------------------------------------------------------------#
'''--------------------------------------------------------------------------'''
# ----------------------------------------------------------------------------#
try:
    import os
    import sys
    import locale
    import time
    from   time      import strftime, strptime, gmtime
    import datetime      as dt
    from   termcolor import colored
    from   colored   import *
    import oauthlib
    import requests
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    have_systemlibs = True
    import multiprocessing
    import pprint
    import mpmath
    from   enum          import Enum
    import numpy             as np
    from   functools     import reduce
    from   numpy.linalg  import norm
    from   matplotlib    import pyplot           as plt
    from   scipy.stats   import logistic, hypsecant
    from  scipy          import signal
    from  scipy.signal   import dlti             as DLTI
    from  scipy.signal   import TransferFunction as TF
    from  networkx       import DiGraph
except ImportError:
    have_systemlibs = False
    print(colored("pyDefinitions>> an error ocurred importing 'system' modules", 'red'))
# ----------------------------------------------------------------------------#
'''---------------------------------------------------------------------------
Direction of signal in Processing Element
    - forward:  [  fowardOperatorName,   fowardOperatorName_]
    - backward: [backwardOperatorName, backwardOperatorName_]

    Note: The underscore signifies the gradient of the
          operator,
          function,
          variable (gradient with respect to the specific variable)
'''

##ONE = TF(1,1,dt=1.0)


class DIRECTION(Enum):
     FWD = 0
     BWD = 1

'''---------------------------------------------------------------------------
Side of Processing Element
    - IN:  [ inputOperatorName,  inputOperatorName_]
    - OUT: [outputOperatorName, outputOperatorName_]
'''

class SIDE(Enum):
     IN  = 0
     OUT = 1


'''---------------------------------------------------------------------------
Section 1: Activation Functions
- These will be callable via the ACTIVATION_FUNCTION dictionary declared below
- Function name convention:
    - forward:  activationFunctionName
    - backward: activationFunctionName_
'''

class pyOperator(object):
    __uid = 0        # This will keep a count of operators used
    def __init__(self, _forwardOperator  = None):

        self.__uid = pyOperator.__uid
        pyOperator.__uid += 1
        self.forwardOperator      = _forwardOperator[0] # forward operator
        self.backwardOperator     = _forwardOperator[1] # forward operator gradient
        self.forwardOperatorName  = _forwardOperator[2] # forward operator gradient
        self.backwardOperatorName = _forwardOperator[3] # forward operator gradient

    def __call__(self, x):
        return self.forwardOperator(x)

    def forward(self, x):
        return self.forwardOperator(x)

    def backward(self, x):
        return self.backwardOperator(x)

    def __str__(self):
        return self.forwardOperatorName

    def __repr__(self):
        return "I am {}: ({},{})".format(self.__class__.__name__,
                                         self.forwardOperatorName,
                                         self.backwardOperatorName)

    def test(self):
        op1 = pyOperator([identity,   identity_,   'identity',   'identity_'])
        op2 = pyOperator([Tanh,       Tanh_,       'tanh',       'tanh_'])
        print("op1(3.5) = ",op1.forward(3.5))
        print("op1.forward(3.5) = ",op1.forward(3.5))
        print("op1.backward(3.5) = ",op1.backward(3.5))
        print("op1 = ",op1)


# Identity
#-----------------------------------------------------
def identity(x):
    return (x)

def identity_(x):
    return (np.ones_like(x))

# Sigmoid
#-----------------------------------------------------
def sigmoid(x):
    return 1./(1. + np.exp(-x))

def sigmoid_(x):
    return sigmoid(x)*(1-sigmoid(x))

# Tanh
#-----------------------------------------------------
def Tanh(x):
    return np.tanh(x)

def Tanh_(x):
    return (1+np.tanh(x))*(1-np.tanh(x))

# RBF
#-----------------------------------------------------
def RBF(x):
    return np.exp(-np.power(np.array(x),2))

def RBF_(x):
    return -2*(x*np.exp(-np.power(np.array(x),2)))

# RBF2
#-----------------------------------------------------
def RBF2(x, c=None):
    if c is None: c = np.zeros_like(x)
    return np.exp(-np.power((np.array(x)-np.array(c)),2))

def RBF2_(x, c=None):
    if c is None: c = np.zeros_like(x)
    return 2*(x-c)*np.exp(-np.power((np.array(x)-np.array(c)),2))

# ReLU
#-----------------------------------------------------
def ReLU(x):
    return x  * (x > 0)

def ReLU_(x):
    return 1. * (x > 0)

# Soft Plus
#-----------------------------------------------------
def softPlus(x, s=1):
    return np.log(1. + np.exp( s*x))

def softPlus_(x, s=1):
    return s/(1. + np.exp(-s*x))

# Log Cosh
#-----------------------------------------------------
def logCosh(x, s=1):
    return np.log(np.cosh(s*x))

def logCosh_(x, s=1):
    return s*np.tanh(s*x)

# Soft ReLU
#-----------------------------------------------------
def softReLU(x):
    return softPlus (x)

def softReLU_(x):
    return softPlus_(x)

# Saturation
#-----------------------------------------------------
def saturation(x):
    r = np.array(x)
    r[abs(r)>1] = np.sign(r[abs(r)>1])
    return r

def saturation_(x):
    r = np.ones_like(x)
    x = np.array(x)
    r[abs(x)>1] = 0
    return r

# Hard limit
#-----------------------------------------------------
def hardlimit(x):
    return 1 * (x > 0)

def hardlimit_(x):
    return 0 * x

# Sign
#-----------------------------------------------------
def Sign(x):
    return np.sign(x)

def Sign_(x):
    return 0 * x

# Sin
#-----------------------------------------------------
def Sin(x):
    # when x is scalar, convert to a list to keep subsequent code simple
    if not hasattr(x, '__iter__'):
        x = [x]
    r = np.sin(np.dot(np.pi/2,x))
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = np.sign(v)
    return r

def Sin_(x):
    # when x is scalar, convert to a list to keep subsequent code simple
    if not hasattr(x, '__iter__'):
        x = [x]
    r = np.pi*np.cos(np.dot(np.pi/2,x))/2
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = 0
    return r

# Cos
#-----------------------------------------------------
def Cos(x):
    # when x is scalar, convert to a list to keep subsequent code simple
    if not hasattr(x, '__iter__'):
        x = [x]
    r = 0.5 * (1. + np.cos(np.dot(np.pi,x)))
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = 0
    return r

def Cos_(x):
    # when x is scalar, convert to a list to keep subsequent code simple
    if not hasattr(x, '__iter__'):
        x = [x]
    r = 0.5*np.pi*np.sin(-np.dot(np.pi,x))
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = 0
    return r

#def softmax(x):
#-----------------------------------------------------
#    e = np.exp(x - np.max(x))  # prevent overflow
#    if e.ndim == 1:
#        return e / np.sum(e, axis=0)
#    else:
#        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

class ActFunIdx(Enum):
     FORWARD  = 0
     GRADIENT = 1
     FWD_NAME = 2
     GRD_NAME = 3

ACTIVATION_FUNCTION = {
    'linear':     (identity,   identity_,   'identity',   'identity_'  ),
    'tanh':       (Tanh,       Tanh_,       'tanh',       'tanh_'      ),
    'sigmoid':    (sigmoid,    sigmoid_,    'sigmoid',    'sigmoid_'   ),
    'relu':       (ReLU,       ReLU_,       'ReLU',       'ReLU_'      ),
    'softplus':   (softPlus,   softPlus_,   'softPlus',   'softPlus_'  ),
    'logCosh':    (logCosh,    logCosh_,    'logCosh',    'logCosh_'   ),
    'softReLU':   (softReLU,   softReLU_,   'softReLU',   'softReLU_'  ),
    'rbf':        (RBF,        RBF_,        'rbf',        'rbf_'       ),
    'rbf2':       (RBF2,       RBF2_,       'rbf2',       'rbf2_'      ),
    'saturation': (saturation, saturation_, 'saturation', 'saturation_'),
    'hardlimit':  (hardlimit,  hardlimit_,  'hardlimit',  'hardlimit_' ),
    'sign':       (Sign,       Sign_,       'sign',       'sign_'      ),
    'sin':        (Sin,        Sin_,        'sin',        'sin_'       ),
    'cos':        (Cos,        Cos_,        'cos',        'cos_'       ),
    }





'''----------------------------------------------------------------------
Section 2: Input Operators
- These will be callable via the INPUT_OPERATOR dictionary declared below
- Function name convention:
    - forward:  inputOperatorName
    - backward: inputOperatorName_
'''

# Product
#-----------------------------------------------------
def Prod(v):
	return reduce(lambda r, e: r*e, v)


INPUT_OPERATOR = {
    'sum':     (np.sum,  identity_, 'sum',     'sum_'    ),
    'product': (Prod,    Tanh_,     'product', 'product_'),
    'max':     (np.max,  sigmoid_,  'max ',    'max_'    ),
    'norm':    (norm,    ReLU_,     'norm',    'norm_'   ),
    'mean':    (np.mean, softPlus_, 'mean',    'mean_'   ),
    }



'''---------------------------------------------------------------------
Section 3: Loss Functions
- These will be callable via the LOSS_FUNCTION dictionary declared below
- Function name convention:
    - forward:  lossFunctionName
    - backward: lossFunctionName_
'''

# LpNorm
#-----------------------------------------------------
def LpNorm(e, p = 2, axis = 1):
    return norm(e,p, axis = axis)

def LpNorm_(e, p = 2, axis = None):
    return norm(np.sign(e)/norm(e,p),p-1)*np.sign(e)

# Quadratic
#-----------------------------------------------------
def quadratic(e):
    return e*e/2. # LpNorm(e,2)

def quadratic_(e):
    return e

# Absolute
#-----------------------------------------------------
def absolute(e):
    return np.abs(e)

def absolute_(e):
    return np.sign(e)

# LogCosh
#-----------------------------------------------------
def LogCosh(e):
    return np.log(np.cosh(e))

def LogCosh_(e):
    return np.tanh(e)

# Skewed soft
#-----------------------------------------------------
def skewSoft(x, s=5):
    return np.log(1. + np.exp( s*x))

def skewSoft_(x, s=5):
    return     s/(1. + np.exp(-s*x))

# Skewed log cosh
#-----------------------------------------------------
def skLogCosh(e, slope = 5, gamma = 1.0):
    gain = np.power(slope,np.tanh(np.dot(gamma,e)))
    cost = np.multiply(gain,LogCosh(e))
    return cost

def sech(x):
    return np.divide(1.,np.cosh(x))
def csch(x):
    return np.divide(1.,np.sinh(x))

def skLogCosh_(e, slope = 5, gamma = 1.0):
    gain = np.power(slope,np.tanh(np.dot(gamma,e)))
#    gradient = gain*(np.tanh(e) + gamma*np.log(slope)*np.power(mpmath.sech(np.dot(gamma,e)),2.))
    gradient = gain*(np.tanh(e) + gamma*np.log(slope)*np.power(hypsecant.pdf(np.dot(gamma,e)),2.))
    return gradient

def skewedLogCosh_(e, slope = 5, gamma = 1.0):
    gain = np.power(slope,np.tanh(np.dot(gamma,e)))
    gradient = gain*(np.tanh(e) + gamma*np.log(slope)*np.power(mpmath.sech(np.dot(gamma,e)),2.))
    return gradient

# Max
#-----------------------------------------------------
def Max(e):
    return max(e) # LpNorm(e,np.inf)

def Max_(e):
    return [int(abs(i)<1) for i in e]

# Log likelihood
#-----------------------------------------------------
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    np.sum( np.multiply(target,scores) - np.log(1. + np.exp(scores)) )
    return ll

# Cross entropy
#-----------------------------------------------------
def crossEntropy(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1.-t), np.log(1.-y)))


LOSS_FUNCTION = {
    'quadratic': (quadratic, quadratic_, 'quadratic', 'quadratic_'),
    'logcosh':   (LogCosh,   LogCosh_,   'logcosh',   'logcosh_'  ),
    'sklogcosh': (skLogCosh, skLogCosh_, 'sklogcosh', 'sklogcosh_'),
    'skewedlogcosh': (skLogCosh, skewedLogCosh_, 'skewlogcosh', 'skewlogcosh_'),
    'softplus':  (softPlus,  softPlus_,  'softPlus',  'softPlus_' ),
    'skewSoft':  (skewSoft,  skewSoft_,  'skewSoft',  'skewSoft_' ),
    'absolute':  (absolute,  absolute_,  'absolute',  'absolute_' ),
    #        'max'      :(Max       ,       Max_, 'Max'       ,       'Max_'),
    #        'LpNorm'   :(LpNorm    ,    LpNorm_, 'LpNorm'    ,    'LpNorm_'),
    }



'''----------------------------------------------------------------------
Section 4: Axon Input Operators
'''

# Product
#-----------------------------------------------------
def product(inData, weight):
    return inData * weight

def product_():
    # ?
    return 0

# Difference
#-----------------------------------------------------
def difference(inData, weight):
    return weight - inData

def difference_(inData, weight):
    return weight - inData

AXON_INPUT_OPERATOR = {
    'product':    (product,    product_,    'product',     'product_'  ),
    'difference': (difference, difference_, 'difference', 'difference_'),
    }



'''----------------------------------------------------------------------
Section 5: Weight Type Initializations
'''

def ones(arrShapeTuple = None):

    if arrShapeTuple == None:
        return 1

    return np.ones(arrShapeTuple)


WEIGHT_TYPE = {
    'uniRand0To1': (np.random.rand, 'uniform random [0, 1)'),
    'ones': (ones, 'ones'),
    }

# Numpy references:
# np.random.rand:  https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html


"""
    ---------------------------------------------------------------------------
    SAMPLING METHOD
    ---------------------------------------------------------------------------
"""
def next_batch(X, Y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], Y[i:i + batchSize])


'''----------------------------------------------------------------------
Example of Definitions
'''
#<------------------------------ main() function for testing if run alone
def testDefinitions():
    plt.rc_context({'axes.edgecolor'  :'red',
                    'figure.facecolor':'white',
                    'font.family'     :'Parallax',
                    'font.weight'     :'heavy',
                  })
    width, height = 10, 10
    xx = np.linspace(-2, 2, 750)#(-5, 5, 250)

    #--------------------------------------------------------------------#
    """ Sample of Activation Functions and It's Derivatives """
    #--------------------------------------------------------------------#
#    fig = plt.figure(1,figsize=(width, height)) # (width, height) [inches]
#    fig.clf()
#    ax = fig.gca()
#    plt.hold(True)
#    for activation in ACTIVATION_FUNCTION:
#        forward, backward, fName, bName = ACTIVATION_FUNCTION[activation]
#        print(fName, bName)
#        yy = [forward(x)  for x in xx]
#        dd = [backward(x) for x in xx]
#        plt.plot(xx, yy, label=fName)
#        plt.plot(xx, dd, '-', label=bName)
#    ax.axis('equal')
#    plt.title("Sample of Activation Functions and It's Derivatives")
#    ax.legend()
#    fig.draw
#    plt.grid(True)
#    plt.hold(False)
#    plt.show()

    #--------------------------------------------------------------------#
    """ Sample of Activation Functions and It's Derivatives """
    #--------------------------------------------------------------------#
    fig = plt.figure(2,figsize=(width, height)) # (width, height) [inches]
    fig.clf()
    ax = fig.gca()
    plt.hold(True)
    testLOSS_FUNCTION = ['sklogcosh', 'skewedlogcosh']
    for loss in testLOSS_FUNCTION:
        forward, backward, fName, bName = LOSS_FUNCTION[loss]
        print(fName, bName)
        yy = [forward(x)  for x in xx]
        dd = [backward(x) for x in xx]
        plt.plot(xx, yy, label=fName)
        plt.plot(xx, dd, '-', label=bName)
#    ax.axis('equal')
    plt.title("Sample of Loss Functions and It's Gradients")
    ax.legend()
    fig.draw
    plt.grid(True)
    plt.hold(False)
    plt.show()

if __name__ == "__main__":
    testDefinitions()





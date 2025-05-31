#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""----------------------------------------------------------------------------
Created for [SPAT, LLC] on %(date)s
@author(s): benito | alejo | juan
         benito@SPATaps.com [benito.fernandez@gmail.com]
(c) 2017 Salute, Physique, Aesthetica, Technologie, LLC
-------------------------------------------------------------------------------
Purpose:
-------
    <Briefly describe the functionality of the code in the file.>
    This code reads a file entered as parameter and returns
    a dataframe with the data.
-------------------------------------------------------------------------------
Source:
------
-------------------------------------------------------------------------------
Parameters:
----------
    <List the paramters that are needed by the code and their meaning...>
    fileName : str (path)
        file name to be loaded by pandas into the dataframe df.
        the filename includes the path to the file (else, it is assumed
        the file exists in the current directory.

-------------------------------------------------------------------------------
Returns:
-------
    <List the variables that are returned and their meaning...>
    df : pandas.DataFrame
        dataFrame of data read from file
-------------------------------------------------------------------------------
Notes:
-----
    <Write here notes to help describe the functionality of the code, like
    what algorithms are used, which modules are needed, etc.>
    This code requires to import the pandas module.
-------------------------------------------------------------------------------
Example:
-------
    <Show examples on how to use the code in different instances.
    For example, explain the use of the default parameters.>
    >>> import pandas as pd
    >>> fileName='./aapl.csv'
    >>> df = getDataFrame(filename=fileName)
    >>> print("The data read from <%s> is:\n"%fileName, df.head())
----------------------------------------------------------------------------"""

'''-------------------------------------------------------------------------'''
import math
import numpy    as np
import pandas   as pd
import scipy    as sp
import datetime as dt
from   scipy         import *
from   scipy.linalg  import norm, pinv
from   scipy.stats   import logistic, hypsecant
import matplotlib        as mpl
from   matplotlib    import pyplot as plt
from   TextFormatter import *
"""-------------------------------------------------------------------------"""
cprint = TextFormatter()
cprint.cfg('y', 'k', 'u')
"""
    ---------------------------------------------------------------------------
    ACTIVATION FUNTIONS
    ---------------------------------------------------------------------------
"""
def     identity (x): return (x)
def     identity_(x): return (np.ones_like(x))

def      sigmoid (x): return 1./(1. + np.exp(-x))        # activation function: sigmoid
def      sigmoid_(x): return sigmoid(x)*(1-sigmoid(x))   # derivative of        sigmoid

def         Tanh (x): return np.tanh(x)                  # activation function: Tanh
def         Tanh_(x): return (1+np.tanh(x))*(1-np.tanh(x))# derivative of       Tanh

def          RBF (x): return np.exp(-np.power(np.array(x),2))
def          RBF_(x): return -2*(x*np.exp(-np.power(np.array(x),2)))

def     rbf2 (x,c=0): return np.exp(-np.power((np.array(x)-np.array(c)),2))
def     rbf2_(x,c=0): return 2*(x-c)*np.exp(-np.power((np.array(x)-np.array(c)),2))

def         ReLU (x): return x  * (x > 0)                # activation function: ReLU
def         ReLU_(x): return 1. * (x > 0)                # derivative of        ReLU

def softPlus (x, s=1): return np.log(1. + np.exp( s*x))
def softPlus_(x, s=1): return    s/(1. + np.exp(-s*x))

def    softReLU (x): return softPlus (x)
def    softReLU_(x): return softPlus_(x)

def  saturation (x):                                    # activation function: saturation
                     r = x
                     r[abs(r)>1] = np.sign(r[abs(r)>1])
                     return r
def  saturation_(x):                                    # derivative of        saturation
                     r = np.ones_like(x)
                     r[abs(x)>1] = 0
                     return r

def   hardlimit (x): return 1 * (x > 0)                 # activation function: hardlimit
def   hardlimit_(x): return 0*x                         # derivative of        hardlimit

def        Sign (x): return np.sign(x)                  # activation function: Sign
def        Sign_(x): return x*0                         # derivative of        Sign

def         Sin (x):                                    # activation function: Sin
                     r = np.sin(np.pi*x/2)
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = np.sign(v)
                     return r
def         Sin_(x):                                    # derivative of        Sin
                     r = np.pi*np.cos(np.pi*x/2)/2
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = 0
                     return r

def         Cos (x):                                    # activation function: Cos
                     r = 0.5*(1.+np.cos(np.pi*x))
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = 0
                     return r
def         Cos_(x):                                    # derivative of        Cos
                     r = 0.5*np.pi*np.sin(-np.pi*x)
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = 0
                     return r

#def softmax(x):
#    e = np.exp(x - np.max(x))  # prevent overflow
#    if e.ndim == 1:
#        return e / np.sum(e, axis=0)
#    else:
#        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

ACTIVATION_FUNCTION = {
        'linear'    :  (identity   ,   identity_, 'identity'  , 'identity_'  ),
        'tanh'      :  (Tanh       ,       Tanh_, 'tanh'      ,     'tanh_'  ),
        'sigmoid'   :  (sigmoid    ,    sigmoid_, 'sigmoid '  ,  'sigmoid_'  ),
        'relu'      :  (ReLU       ,       ReLU_, 'ReLU'      ,     'ReLU_'  ),
        'softplus'  :  (softPlus   ,   softPlus_, 'softPlus'  ,   'softPlus_'),
        'softReLU'  :  (softReLU   ,   softReLU_, 'softReLU'  ,   'softReLU_'),
        'rbf'       :  (RBF        ,        RBF_, 'rbf'       ,      'rbf_'  ),
        'saturation':  (saturation , saturation_, 'saturation', 'saturation_'),
        'hardlimit' :  (hardlimit  ,  hardlimit_, 'hardlimit' , 'hardlimit_' ),
        'sign'      :  (Sign       ,       Sign_, 'sign'      , 'sign_'      ),
        'sin'       :  (Sin        ,        Sin_, 'sin'       , 'sin_'       ),
        'cos'       :  (Cos        ,        Cos_, 'cos'       , 'cos_'       ),
        }

"""
    ---------------------------------------------------------------------------
    LOSS/COST FUNTIONS
    ---------------------------------------------------------------------------
"""
def  LpNorm (e, p=2, axis = 1):
                    return norm(e,p, axis = axis)
def  LpNorm_(e, p=2, axis = None):
                    return norm(np.sign(e)/norm(e,p),p-1)*np.sign(e)

def quadratic (e):  return e*e/2. # LpNorm(e,2)
def quadratic_(e):  return e

def  absolute (e):  return np.abs(e)
def  absolute_(e):  return np.sign(e)

def   LogCosh (e):  return np.log(np.cosh(e))      # cost function: LogCosh
def   LogCosh_(e):  return np.tanh(e)              # gradient of    LogCosh

def skewSoft (x, s=1): return np.log(1. + np.exp( s*x))
def skewSoft_(x, s=1): return    s/(1. + np.exp(-s*x))

def  skLogCosh (e, slope = 5, gamma = 2):
                    gain = np.power(slope,np.tanh(gamma*e))
                    cost = gain*LogCosh(e)
                    return cost                    # cost function: skLogCosh
def  skLogCosh_(e, slope = 5, gamma = 2):
                    gain = np.power(slope,np.tanh(gamma*e))
                    gradient = gain*(np.tanh(e) \
                             + gamma*np.log(slope)*np.power(hypsecant.pdf(gamma*e),2.))
                    return gradient                # gradient of    skLogCosh

def       Max (e):  return max(e) # LpNorm(e,np.inf)
def       Max_(e):  return [int(abs(i)<1) for i in e]

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1. + np.exp(scores)) )
    return ll

def crossEntropy(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1.-t), np.log(1.-y)))


LOSS_FUNCTION = {
        'quadratic':(quadratic , quadratic_, 'quadratic' , 'quadratic_'),
        'logcosh'  :(LogCosh   ,   LogCosh_, 'logcosh'   ,   'logcosh_'),
        'sklogcosh':(skLogCosh , skLogCosh_, 'sklogcosh' , 'sklogcosh_'),
        'softplus' :(softPlus  ,  softPlus_, 'softPlus'  ,  'softPlus_'),
        'skewSoft' :(skewSoft  ,  skewSoft_, 'skewSoft'  ,  'skewSoft_'),
        'absolute' :(absolute  ,  absolute_, 'absolute'  ,  'absolute_'),
#        'max'      :(Max       ,       Max_, 'Max'       ,       'Max_'),
#        'LpNorm'   :(LpNorm    ,    LpNorm_, 'LpNorm'    ,    'LpNorm_'),
        }

'''-------------------------------------------------------------------------'''

plt.rc_context({'axes.edgecolor'  :'red',
                'figure.facecolor':'white',
                'font.family'     :'Parallax',
                'font.weight'     :'heavy',
                'interactive'     : True,
                'axis.grid'       : True,
              })

'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''

Xmin, Xmax, Xstep = -3, 3.01, .01

def plotAllActivationsTogether():
    plt.figure()
    # plt.clf()
    cprint.out('plotting activation functions')
    leyenda = list()
    for activ in ACTIVATION_FUNCTION.keys():
        x = np.arange(Xmin, Xmax, Xstep)
        print('plotting activation function: ', activ)
        y = ACTIVATION_FUNCTION[activ][0](x)
        plt.plot(x,y)
        leyenda.append(ACTIVATION_FUNCTION[activ][2])
    plt.legend(leyenda)
    plt.title("Testing Activation Functions - All")
    plt.grid(True)
    plt.show()
    plt.savefig('pictures/ActivationFunctions-All.png')


'''-------------------------------------------------------------------------'''

def plotActivation(activation=-1):
    x = np.arange(Xmin, Xmax, Xstep)
    activ = ACTIVATION_FUNCTION[activation]
    print(activ)
    plt.figure()
    # plt.clf()
    print('plotting activation function: ', activ)
    y = ACTIVATION_FUNCTION[activ][0](x)
    g = ACTIVATION_FUNCTION[activ][1](x)
    plt.plot(x,y,'-',x,g,'--')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([Xmin, Xmax, ymin, ymax])
    plt.legend([ACTIVATION_FUNCTION[activ][2], ACTIVATION_FUNCTION[activ][3]])
    plt.title("Testing Activation Function: "+activ)
    plt.grid(True)
    # plt.show()
    plt.savefig("pictures/ActivationFunction-"+activ+'.png')


def plotAllActivations(activation=-1):
    cprint.out('Plotting individual activation functions')
    keys = list(ACTIVATION_FUNCTION.keys())
    vals = list(ACTIVATION_FUNCTION.values())
    if activation == -1:
        for activ in keys:
            plotActivation(activ)
            plt.savefig("pictures/ActivationFunction-"+vals[activ]+'.png')
    else:
        plotActivation(activation)
        plt.savefig("pictures/ActivationFunction-"+vals[activation]+'.png')

##def plotAllActivations(activation=-1):
##    cprint.out('Plotting individual activation functions')
##    for activ in ACTIVATION_FUNCTION.keys():
##    x = np.arange(Xmin, Xmax, Xstep)
##    plt.figure()
##    # plt.clf()
##    print('plotting activation function: ', activ)
##    y = ACTIVATION_FUNCTION[activ][0](x)
##    g = ACTIVATION_FUNCTION[activ][1](x)
##    plt.plot(x,y,'-',x,g,'--')
##    xmin, xmax, ymin, ymax = plt.axis()
##    plt.axis([Xmin, Xmax, ymin, ymax])
##    plt.legend([ACTIVATION_FUNCTION[activ][2], ACTIVATION_FUNCTION[activ][3]])
##    plt.title("Testing Activation Function: "+activ)
##    plt.grid(True)
##    # plt.show()
##    plt.savefig("pictures/ActivationFunction-"+activ+'.png')
##
#ACTIVATION_FUNCTION = {
#        'linear'    :  (identity   ,   identity_, 'identity'  , 'identity_'  ),
#        'tanh'      :  (Tanh       ,       Tanh_, 'tanh'      ,     'tanh_'  ),
#        'sigmoid'   :  (sigmoid    ,    sigmoid_, 'sigmoid '  ,  'sigmoid_'  ),
#        'relu'      :  (ReLU       ,       ReLU_, 'ReLU'      ,     'ReLU_'  ),
#        'rbf'       :  (RBF        ,        RBF_, 'rbf'       ,      'rbf_'  ),
#        'logistic'  :  (Logistic   ,   Logistic_, 'logistic'  , 'logistic_'  ),
#        'saturation':  (saturation , saturation_, 'saturation', 'saturation_'),
#        'hardlimit' :  (hardlimit  ,  hardlimit_, 'hardlimit' , 'hardlimit_' ),
#        'sign'      :  (Sign       ,       Sign_, 'sign'      , 'sign_'      ),
#        'sin'       :  (Sin        ,        Sin_, 'sin'       , 'sin_'       ),
#        'cos'       :  (Cos        ,        Cos_, 'cos'       , 'cos_'       ),
#        }

'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''

def plotAllLossFunctionsTogether():
    plt.figure()
    # plt.clf()
    leyenda = list()
    cprint.out('plotting loss functions')
    for loss in LOSS_FUNCTION.keys():
        e = np.arange(Xmin, Xmax, Xstep)
        print('plotting loss function: ', loss)
        c = LOSS_FUNCTION[loss][0](e)
        plt.plot(e,c)
        plt.savefig(loss+'_LossFunction.png')
        leyenda.append(LOSS_FUNCTION[loss][2])
    e = np.arange(Xmin, Xmax, Xstep)
    c = LOSS_FUNCTION['sklogcosh'][0](e, slope = 1./5)
    plt.plot(e,c,label=LOSS_FUNCTION['sklogcosh'][0])
    plt.legend(leyenda)
    plt.title("Testing Cost/Loss Functions - All")
    plt.grid(True)
    # plt.show()
    plt.savefig('pictures/LossFunctions-All.png')

'''-------------------------------------------------------------------------'''

def plotAllLossFunctions():
    cprint.out('plotting individual cost functions')
    for loss in LOSS_FUNCTION.keys():
        x = np.arange(Xmin, Xmax, Xstep)
        plt.figure()
        # plt.clf()
        print('plotting cost function: ', loss)
        y = LOSS_FUNCTION[loss][0](x)
        g = LOSS_FUNCTION[loss][1](x)
        plt.plot(x,y,'-',x,g,'--')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.axis([Xmin, Xmax, ymin, ymax])
        plt.legend([LOSS_FUNCTION[loss][2], LOSS_FUNCTION[loss][3]])
        plt.title("pictures/"+loss+"_LossFunction: ")
        plt.grid(True)
        # plt.show()
        plt.savefig("pictures/LossFunction-"+loss+'.png')

#LOSS_FUNCTION = {
#        'quadratic':(quadratic , quadratic_, 'quadratic' , 'quadratic_'),
#        'logcosh'  :(LogCosh   ,   LogCosh_, 'logcosh'   ,   'logcosh_'),
#        'absolute' :(absolute  ,  absolute_, 'absolute'  ,  'absolute_'),
#        'sklogcosh':(skLogCosh , skLogCosh_, 'sklogcosh' , 'sklogcosh_'),
#        'relu'     :(ReLU      ,      ReLU_, 'ReLU'      ,      'ReLU_'),
##        'LpNorm'   :(LpNorm    ,    LpNorm_, 'LpNorm'    ,    'LpNorm_'),
#        }

'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''

def plotLogCoshLossFunctionSlope():
    plt.figure()
    # plt.clf()
    leyenda = list()
    for s in reversed(range(2,9)):
        c = LOSS_FUNCTION['sklogcosh'][0](e, slope = s)
        plt.plot(e,c)
        leyenda.append("%.2f"%(s))

    c = LOSS_FUNCTION['sklogcosh'][0](e, slope = 1)
    plt.plot(e,c,'.')
    leyenda.append("%.2f"%(1.))

    for s in range(2,9):
        c = LOSS_FUNCTION['sklogcosh'][0](e, slope = 1./s)
        plt.plot(e,c, '--')
        leyenda.append("%.2f"%(1./s))
    plt.legend(leyenda)
    plt.title("Testing Skewed LogCosh Cost/Loss Function - varying slope")
    plt.grid(True)
    # plt.show()
    plt.savefig('pictures/LossFunction-Skewed[varying slope].png')

'''-------------------------------------------------------------------------'''

def plotLogCoshLossFunctionGamma():
    #with mpl.rc_context(rc={}, fname='matplotlibrc/ggplot2/ggplot2_paper'):
    plt.figure()
    # plt.clf()
    leyenda = list()
    for g in reversed(range(2,9)):
        e = np.arange(-1, 1.01, .01)
        c = LOSS_FUNCTION['sklogcosh'][0](e, gamma = g)
        plt.plot(e,c)
        leyenda.append("gamma = %.2f"%(g))
    e = np.arange(-1, 1.01, .01)
    c = LOSS_FUNCTION['sklogcosh'][0](e, gamma = 1)
    plt.plot(e,c,'.')
    leyenda.append("gamma = %.2f"%(1.))

    for g in range(2,9):
        e = np.arange(-1, 1.01, .01)
        c = LOSS_FUNCTION['sklogcosh'][0](e, gamma = 1./g)
        plt.plot(e,c, '--')
        leyenda.append("gamma = %.2f"%(1./g))
    plt.legend(leyenda)
    plt.title("Testing Skewed Cost/Loss Function - varying gamma")
    plt.grid(True)
    # plt.show()
    plt.savefig('pictures/LossFunction-Skewed[varying gamma].png')

'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''

def plotLogCoshLossFunctionBoth():
    plt.figure()
    # plt.clf()
    leyenda = list()
    for s in reversed(range(2,9)):
        e = np.arange(Xmin, Xmax, Xstep)
        c = LOSS_FUNCTION['sklogcosh'][0](e, slope = 1./s)
        plt.plot(e,c)
        leyenda.append("alpha=%.2f"%(s))

    c = LOSS_FUNCTION['sklogcosh'][0](e, slope = 10, gamma = 1)
    plt.plot(e,c,'.')
    leyenda.append("alpha=%.2f, gamma=%.2f"%(10,1.))

    for g in range(2,9):
        e = np.arange(Xmin, Xmax, Xstep)
        c = LOSS_FUNCTION['sklogcosh'][0](e, slope = 10, gamma = 1./g)
        plt.plot(e,c, '--')
        leyenda.append("gamma=%.2f"%(g))
    plt.legend(leyenda)
    plt.title("Testing Skewed Cost/Loss Function - varying slope & gamma")
    plt.grid(True)
    plt.show()
    plt.savefig('pictures/LossFunction-Skewed[varying slope & gamma].png')

'''-------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------'''

def select_option(options, prompt="Choose an option: "):
    # Display the options
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    # Ask the user to choose an option
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(options):
                return choice
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    # Sample list of options
    opciones = ["plot ALL", \
                "plot Activation Functions (1 plot)", \
                "plot Activation Functions (separately)", \
                "plot Loss Functions (1 plot)", \
                "plot Loss Functions (separately)", \
                "plot LogCosh Loss Function - varying slope", \
                "plot LogCosh Loss Function - varying gamma", \
                "plot LogCosh Loss Function - varying both"]

    # Call the function and get the chosen option number
    option = select_option(opciones)
    print(f"You chose option number: {option}")

    print(50*'*')
    print("  Welcome to Benito Fernandez's Testing Facility")
    print(50*'*')
    print("\nPlease select an option:")
    max_length = max(len(s) for s in ACTIVATION_FUNCTION)
    print("Max length ", max_length)
    for a in ACTIVATION_FUNCTION:
        print(f'{a:<{max_length}}', ": ", ACTIVATION_FUNCTION[a][2], ACTIVATION_FUNCTION[a][3])

    if option == 1:
        plotAllActivationsTogether()
        plotAllActivations()
        plotAllLossFunctionsTogether()
        plotAllLossFunctions()
        plotLogCoshLossFunctionSlope()
        plotLogCoshLossFunctionGamma()
        plotLogCoshLossFunctionBoth()
    elif option == 2:
        plotAllActivationsTogether()
    elif option == 3:
        plotAllActivations()
    elif option == 4:
        plotAllLossFunctionsTogether()
    elif option == 5:
        plotAllLossFunctions()
    elif option == 6:
        plotLogCoshLossFunctionSlope()
    elif option == 7:
        plotLogCoshLossFunctionGamma()
    elif option == 8:
        plotLogCoshLossFunctionBoth()
    else:
        print("Done!")


if __name__ == '__main__':
    main()

'''-------------------------------------------------------------------------'''
"""-------                        END-OF-FILE                        -------"""
'''-------------------------------------------------------------------------'''

#with mpl.rc_context(rc={'interactive': False}):
#    fig, ax = plt.subplots()
#    ax.plot(range(3), range(3))
#    fig.savefig('A.png', format='png')
#    plt.close(fig)


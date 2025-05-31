#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""
"""
@author: benito
(c) 2017 SPAT, LLC
         Salute Physique Aesthetica Technolgie, LLC
         Austin, Texas, USA - The City of Ideas
"""
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    from  pyDefinitions       import INPUT_OPERATOR
    from  pyProcessingElement import ProcessingElement
else:
    from .pyDefinitions       import INPUT_OPERATOR
    from .pyProcessingElement import ProcessingElement
"""-------------------------------------------------------------------------"""

#import math
#import numpy as np

# Base Node object
class Synapse(ProcessingElement):

    __synapsesCreated = 0        # This will keep a count of Synapses created
    '''
    - Parameters:
                - variable: parameter that depends on derived class
                            for Soma (NNet Node), variable stores the Soma's bias
                            for Axon (NNet Link), variable stores the Axon's weight
                - inputOperator: operator that depends on derived class
                            for Soma (NNet Node), operator is the Soma's inputOperator
                            for Axon (NNet Link), operator is the Axon's inputOperator
                - outputOperator: operator that depends on derived class
                            for Soma (NNet Node), operator is the Soma's Activation Function
                            for Axon (NNet Link), operator is the Axon's outputOperator
    '''
    def __init__(self,
             _id   = None,
             _type = None,
             _name = None,
             _forwardOperator  = INPUT_OPERATOR['sum'],
             _backwardOperator = None,
             _operatorKey = None, # input operator default = dot-product
             _inputOperator  = INPUT_OPERATOR['sum'],
             _weights = None,     # weights of links (edges) from input neurons
             _bias    = None,     # bias of neuron
             _neuron  = None,     # owning neuron
             _layer   = None,     # layer   that owns its neuron
             _ganglia = None):    # ganglia that owns its layer

                Synapse.__synapsesCreated += 1
                super(Synapse, self).__init__(_type=_type, _name=_name,
                     forwardOperator  = _forwardOperator,
                     backwardOperator = _backwardOperator)
                self._id      = _id
                self.neuron   = _neuron
                self.layer    = _layer
                self.ganglia  = _ganglia

                self.weights  = _weights
                self.bias     = _bias
                self.variable  = None # internal (state) variable that
                #                       aggregates the input from dendrites (usually the output)
                self.gradients = None # internal (states) gradients with respect to the inputs

                if _operatorKey in INPUT_OPERATOR:
                    self.operator = INPUT_OPERATOR[_operatorKey][0] # input operator
                    self.gradient = INPUT_OPERATOR[_operatorKey][1] # input gradient
                    self.operName = INPUT_OPERATOR[_operatorKey][2] # input operator name
                    self.gradName = INPUT_OPERATOR[_operatorKey][3] # input gradient name

                self.out  = 0    # Will hold output   (returned in forward  method)
                self.grad = 0    # Will hold gradient (returned in backward method)

    def __repr__(self):
        s = super(Synapse, self).__repr__()
#        s += ' :: I am a {name}'.format(name=self.__class__.__name__)
        return s

    def get_SynapsesCreated(self):
        return self.__synapsesCreated

    ## Setters ----------------------------------------

    def setInputOperator(self, inputOperator):
                self.inputOperator     = inputOperator[0]
                self.inputgradient = inputOperator[1] # input operator gradient
                self.inputOperatorName = inputOperator[2]
                self.inputgradientName = inputOperator[3] # input operator gradient

    def setOutputOperator(self, outputOperator):
                self.outputOperator     = outputOperator[0]
                self.outputgradient = outputOperator[1] # output operator gradient
                self.outputOperatorName = outputOperator[2]
                self.outputgradientName = outputOperator[3] # output operator gradient

    def setVar(self, variable):
                self.variable = variable

    ## Getters ----------------------------------------

    def getInputOperator(self, inputOperator):
                return (self.inputOperatorName , self.inputgradientName) # input operator gradient

    def getOutputOperator(self, outputOperator):
                return (self.outputOperatorName , self.outputgradientName) # output operator gradient

    def getVar(self):
                return self.variable


'''----------------------------------------------------------------------
Example of pySynapse
'''
def printMe(d):
    print(30*'-')
    print(d)
    print('_type:      ', d._type,
          '\n_name:    ', d._name,
          '\n_id:      ', d._id,
          '\n bias:    ', d.bias,
          '\n weights: ', d.weights,
          '\n operator:', d.operName,
          '\n gradient:', d.gradName)

#<------------------------------ main() function for testing if run alone
def testpySynapse():
    d = Synapse(_operatorKey='sum', _bias=0.1, _weights = [0.2, 0.3])
    printMe(d)




if __name__ == "__main__":
    testpySynapse()

# --- END OF FILE ---


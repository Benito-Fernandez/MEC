#import math
#import numpy as np
if __name__ != "__main__":
    from .pyObject import pyObject
else:
    from  pyObject import pyObject

# Base Node object
class Node(pyObject):

    __nodesCreated = 0        # This will keep a count of nodes created
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
              _id = None,
              variable = None,
              inputOperator = None,
              outputOperator = None,
              _network=None,
              _type=None,
              _name=None):

                Node.__nodesCreated += 1
                super(Node, self).__init__(_network=_network, _type=_type, _name=_name)
                if _id != None:
                    self._id = _id

                if inputOperator != None:
                    self.inputOperator     = inputOperator[0]
                    self.inputOperatorGrad = inputOperator[1] # input operator gradient

                if outputOperator != None:
                    self.outputOperator     = outputOperator[0]
                    self.outputOperatorGrad = outputOperator[1] # output operator gradient

                self.variable = variable

                self.state  = 0    # Will hold output of inputOperator
                self.output = 0    # Will hold output of outputOperator

    def __repr__(self):
        s = super(Node, self).__repr__()
#        s += ' :: I am a {name}'.format(name=self.__class__.__name__)
        return s

    def get_nodesCreated(self):
        return self.__nodesCreated

    ## Setters ----------------------------------------

    def setInputOperator(self, inputOperator):
                self.inputOperator     = inputOperator[0]
                self.inputOperatorGrad = inputOperator[1] # input operator gradient
                self.inputOperatorName = inputOperator[2]
                self.inputOperatorGradName = inputOperator[3] # input operator gradient

    def setOutputOperator(self, outputOperator):
                self.outputOperator     = outputOperator[0]
                self.outputOperatorGrad = outputOperator[1] # output operator gradient
                self.outputOperatorName = outputOperator[2]
                self.outputOperatorGradName = outputOperator[3] # output operator gradient

    def setVar(self, variable):
                self.variable = variable

    ## Getters ----------------------------------------

    def getInputOperator(self, inputOperator):
                return (self.inputOperatorName , self.inputOperatorGradName) # input operator gradient

    def getOutputOperator(self, outputOperator):
                return (self.outputOperatorName , self.outputOperatorGradName) # output operator gradient

    def getVar(self):
                return self.variable



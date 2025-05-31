#import math
#import numpy as np
"""-------------------------------------------------------------------------"""
if __name__ != "__main__":
    from .pyDefinitions import INPUT_OPERATOR
    from .pyObject      import pyObject, xDebug
else:
    from  pyObject      import pyObject, xDebug
    from  pyDefinitions import INPUT_OPERATOR
"""-------------------------------------------------------------------------"""

# Base ProcessingElement object
class ProcessingElement(pyObject):

    __uid = 0        # This will keep a count of nodes created
    '''
    - Parameters:
                - variable: parameter that depends on derived class
                            for Soma (NNet ProcessingElement), variable stores the Soma's bias
                            for Axon (NNet Link), variable stores the Axon's weight
                - forwardOperator: operator that depends on derived class
                            for Soma (NNet ProcessingElement), operator is the Soma's forwardOperator
                            for Axon (NNet Link), operator is the Axon's forwardOperator
                - backwardOperator: operator that depends on derived class
                            for Soma (NNet ProcessingElement), operator is the Soma's Activation Function
                            for Axon (NNet Link), operator is the Axon's backwardOperator
    '''
    def __init__(self,
              _id      = None,
              _type    = None,
              _name    = None,
              _creator = None,
              forwardOperator  = None):
#              _neuron  = None,     # owning neuron
#              _layer   = None,     # layer   that owns its neuron
#              _ganglia = None,     # ganglia that owns its layer

                self.__uid = ProcessingElement.__uid
                ProcessingElement.__uid += 1
                if _id:
                    self.id = _id
                else:
                    self.id = ProcessingElement.__uid
                ProcessingElement.__uid += 1

                # assign creator
                if _creator:
                    self._creator = _creator
                else:
                    self._creator = None
                self._id = id(self)
                super(ProcessingElement, self).__init__(_type=_type, _name=_name,_creator=self)
                if _id != None:
                    self.id = _id
                if xDebug and _creator is None: print("Creating a pyProcessingElement({}) [{}].".format(self._id,self.id))

                if  forwardOperator is None:
                    forwardOperator = INPUT_OPERATOR['sum']
                self._forwardOperator      = forwardOperator[0] # forward operator
                self._backwardOperator     = forwardOperator[1] # forward operator gradient
                self._forwardOperatorName  = forwardOperator[2] # forward operator name
                self._backwardOperatorName = forwardOperator[3] # forward operator name

                self.variable = None # Will hold variable after fowardOperator
                self.gradient = None # Will hold variable after backwardOperator

    def get_elementsCreated(self):
        return self.__uid

    ## Setters ----------------------------------------

    def setOperator(self, forwardOperator):
                self._forwardOperator      = forwardOperator[0] # forward operator
                self._backwardOperator     = forwardOperator[1] # forward operator gradient
                self._forwardOperatorName  = forwardOperator[2] # forward operator name
                self._backwardOperatorName = forwardOperator[3] # forward operator name

    def setVar(self, variable = 0):
                self.variable = variable

    ## Getters ----------------------------------------

    def getforwardOperator(self):
                return (self._forwardOperatorName)     # input operator

    def getbackwardOperator(self):
                return (self._backwardOperatorName) # input operator gradient

    def getVar(self):
                return self.variable

    def getGradient(self):
                return self.gradient

    def printMe(self):
        print(30*'-')
        print(d)
        print(' _type:            ', self._type,
              '\n _name:            ', self._name,
              '\n _id:              ', self.id,
              '\n _creator:         ', self._creator,
              '\n forwardOperator:  ', self._forwardOperatorName,
              '\n backwardOperator: ', self._backwardOperatorName,
              '\n variable:         ', self.variable,
              '\n gradient:         ', self.gradient)

'''----------------------------------------------------------------------
Example of pyProcessingElement
'''
def printMe(d):
    print(30*'-')
    print(d)
    print(' _type:            ', d._type,
          '\n _name:            ', d._name,
          '\n _id:              ', d._id,
          '\n _creator:         ', d._creator,
          '\n forwardOperator:  ', d._forwardOperatorName,
          '\n backwardOperator: ', d._backwardOperatorName,
          '\n variable:         ', d.variable,
          '\n gradient:         ', d.gradient)

#<------------------------------ main() function for testing if run alone
def testpyProcessingElement():
    d = ProcessingElement()
    printMe(d)

    d2 = ProcessingElement(forwardOperator=INPUT_OPERATOR['product'], _name='FullPE')
    d2.setVar(variable = 1.2)
    printMe(d2)


if __name__ == "__main__":
    testpyProcessingElement()

# --- END OF FILE ---


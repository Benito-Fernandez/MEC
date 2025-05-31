"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    from  pyObject       import pyObject
    from  pyDefinitions  import SIDE, DIRECTION
else:
    from .pyObject       import pyObject
    from .pyDefinitions  import SIDE, DIRECTION
"""-------------------------------------------------------------------------"""

#import math
#import numpy as np

# Base Node object
class Dendrite(pyObject):

    __dendritesCreated = 0        # This will keep a count of dendrites created
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
              _side = SIDE.IN):

            Dendrite.__dendritesCreated += 1
            super(Dendrite, self).__init__(_type=_type, _name=_name)
            if _id != None:
                self._id = _id
            self._side = _side
            self.dendrites = []

    def __str__(self):
        s = super(Dendrite, self).__repr__()
        return s

    def __repr__(self):
        return "pyDendrite[{}]".format(self._id)

    def get_dendritesCreated(self):
        return self.__dendritesCreated

    def connect(self, neurons = []):
        if type(neurons) is list:
            for neuron in neurons:
                self.dendrites.append(neuron)
        elif type(neurons) is self._type:
            print(neurons)
            self.dendrites.append(neurons)



'''----------------------------------------------------------------------
Example of pyDendrite
'''
def printMe(d):
    print(30*'-')
    print(d)
    print('_type:     ', d._type,
          '\n_name:     ', d._name,
          '\n_id:       ', d._id,
          '\n dendrites:', d.dendrites)

#<------------------------------ main() function for testing if run alone
def testpyDendrite():
    print("\n CREATING DENDRITES ...\n")
    d1 = Dendrite()
    printMe(d1)

    d2 = Dendrite(_name='fullDendrite', _type = 'typeGiven')
    printMe(d2)

    d3 = Dendrite(_id=33, _name='fullDendrite2')
    printMe(d3)

    print("\n CONNECTING NEURONS (AXONS) ...\n")

    d1.connect(d2)
    d2.connect([d1,d3])
    d3.connect(neurons=[d3,d2,d1])
    printMe(d1)
    printMe(d2)
    printMe(d3)



if __name__ == "__main__":
    testpyDendrite()

# --- END OF FILE ---


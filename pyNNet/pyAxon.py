"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    from  πDefinitions       import ACTIVATION_FUNCTION, SIDE, DIRECTION
    from  πProcessingElement import ProcessingElement
else:
    from .πDefinitions       import ACTIVATION_FUNCTION, SIDE, DIRECTION
    from .πProcessingElement import ProcessingElement
"""-------------------------------------------------------------------------"""

class ImmutableAxon(ProcessingElement):

    def __init__(self,
              _id   = None,
              _type = None,
              _name = None,
              forwardOperator  = ACTIVATION_FUNCTION['linear']):

        super(ImmutableAxon, self).__init__(_id, _type, _name,
             forwardOperator = forwardOperator)

    def __repr__(self):
        s = super(ImmutableAxon, self).__repr__()
        return s

"""-------------------------------------------------------------------------"""

class Axon(ImmutableAxon):

    __axonsCreated = 0        # This will keep a count of Axons created

    def __init__(self,
              _id   = None,
              _type = None,
              _name = None,
              forwardOperator   = ACTIVATION_FUNCTION['linear']):
        self.forwardOperator    = forwardOperator
        self.activationFunc     = forwardOperator[0] # activation function
        self.activationFuncGrad = forwardOperator[1] # activation function gradient
        self._id = _id

        super(Axon, self).__init__(self._id, _type, _name,
             forwardOperator = forwardOperator)
#        if self.inputOperator not in AXON_INPUT_OPERATOR:
#            raise ValueError('πNet::Axon: input operator not supported.')
        self.potential  = 0
        self.activation = 0
        Axon.__axonsCreated += 1
        self.dendrites = {SIDE.IN: [], SIDE.OUT: []}

    def __repr__(self):
        return "πAxon[{}]".format(self._id)

    def __str__(self):
        s = super(Axon, self).__repr__()
        return s

    def get_axonsCreated(self):
        return self.__axonsCreated

    ## Setter
    def setOperator(self, operator = INPUT_OPERATOR['sum']):
        super(Axon, self).setOperator(operator)

    ## Getter
    def getOperator(self):
        return self.forwardOperator

    def connect(self, port = SIDE.IN, neurons = []):
        if type(neurons) is list:
            for neuron in neurons:
                self.dendrites[port].append(neuron)
        elif type(neurons) is self._type:
            print(neurons)
            self.dendrites[port].append(neurons)

    def getArcs(self, port = SIDE.IN):
        arcs = []
        for neuron in self.dendrites[port]:
            arcs.append(neuron._id)
        return arcs

    ## Network Traversion
    def feedForward(self, inputPotential):
        self.potential  = inputPotential
        self.activation = self.activationFunc(self.potential)

    def backwardPropagation(self, inDataList):
        self.potential_  = self.inputOperatorGrad(inDataList)
        self.activation_ = self.activationFuncGrad(self.potential)

    def backwardPropagation(self, activationGradient = None):

        if self.activationGradient == None:
            self.a_ = 0.0
            for neuron in self.dendrites:
                self.a_ += neuron.getInputGradient()

        self.potential_  = self.inputOperatorGrad(inDataList)
        self.activation_ = self.activationFuncGrad(self.potential)

'''----------------------------------------------------------------------
Example of πAxon
'''
def printMe(d):
    print(30*'-')
    print(d)
    print(' _type:              ', d._type,
          '\n _name:              ', d._name,
          '\n _id:                ', d._id,
          '\n activationFunc:     ', d.forwardOperator[2],
          '\n activationFuncGrad: ', d.forwardOperator[3],
          '\n potential:          ', d.potential,
          '\n activation:         ', d.activation,
          '\n dendrites:          ', d.dendrites)

#<------------------------------ main() function for testing if run alone
def testπAxon():
    print("\n CREATING NEURONS (AXONS) ...\n")
    a1 = Axon()
    printMe(a1)

    a2 = Axon(forwardOperator=ACTIVATION_FUNCTION['tanh'], _name='FullAxon')
    a2.feedForward(inputPotential = 1.2)
    printMe(a2)

    a3 = Axon(forwardOperator=ACTIVATION_FUNCTION['sigmoid'], _name='FullAxon2')
    a3.feedForward(inputPotential = 0.7)
    printMe(a3)

    a4 = Axon(forwardOperator=ACTIVATION_FUNCTION['rbf'], _name='FullAxon3')
    a4.feedForward(inputPotential = [0.7, -0.4, 1.2])
    printMe(a4)

    print("\n CONNECTING NEURONS (AXONS) ...\n")
    a2.connect(a1)
    a1.connect(port=SIDE.OUT,neurons=[a2,a3])
    a2.connect(SIDE.IN,[a3,a1])
    a3.connect(SIDE.OUT,[a2,a1])
    a3.connect(SIDE.IN,[a2,a4])
    a4.connect(SIDE.OUT,[a3,a4])
    printMe(a1)
    printMe(a2)
    printMe(a3)
    printMe(a4)


if __name__ == "__main__":
    testπAxon()

# --- END OF FILE ---



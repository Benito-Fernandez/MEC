"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    from  pyDefinitions       import ACTIVATION_FUNCTION, INPUT_OPERATOR, SIDE, DIRECTION
    from  pyNode              import Node
    from  pyObject            import pyObject
    from  pyProcessingElement import ProcessingElement
else:
    from .pyDefinitions       import ACTIVATION_FUNCTION, INPUT_OPERATOR, SIDE, DIRECTION
    from .pyNode              import Node
    from .pyObject            import pyObject
    from .pyProcessingElement import ProcessingElement
"""-------------------------------------------------------------------------"""

from  scipy         import signal
from  scipy.signal  import dlti   as DLTI
from  scipy.signal  import TransferFunction   as TF
"""-------------------------------------------------------------------------"""

ONE = TF(1,1,dt=1.0)

"""_________________________________________________________________________"""
"""_________________________________________________________________________"""
"""_________________________________________________________________________"""

class Neuron(ProcessingElement):

    __neuronsCreated = 0        # This will keep a count of neurons created

    def __init__(self,
             _id   = None,
             _type = None,
             _name = None,
             _transferFunction = None, # transfer function, default = 1
             _neuron  = None,     # owning neuron
             _layer   = None,     # layer   that owns its neuron
             _ganglia = None,     # ganglia that owns its layer
             inputOperator  = INPUT_OPERATOR['sum'],
             ctivationFunc = ACTIVATION_FUNCTION['linear']):
        '''
        Parameters:
        - _id: tuple, comprised of (layer id, node index in layer, network index)
        '''

        Neuron.__neuronsCreated += 1
        super(Neuron, self).__init__(_id, _type, _name)
        if _id != None:
            self._id = _id

        # Assigning to new variable names (keeps the same memory address)
#        if self.outputOperator in ACTIVATION_FUNCTION:
#            self.activationFunc     = self.outputOperator     # activation function
#            self.activationFuncGrad = self.outputOperatorGrad # activation function gradient
#        else:
#            raise ValueError('pyNet::neuron: activation function not supported.')
        self.activationFunc     = self.outputOperator     # activation function
        self.activationFuncGrad = self.outputOperatorGrad # activation function gradient
        self.bias       = self.variable
        self.potential  = self.state
        self.activation = self.output

    def __repr__(self):
        s = super(Neuron, self).__repr__()
        s += ' -> I am a {name}'.format(name=self.__class__.__name__)
        return s

    def get_neuronsCreated(self):
        return self.__neuronsCreated

    ## Setters
    def setActivationFunc(self, actFunc):
        if actFunc in ACTIVATION_FUNCTION:
            self.setOutputOperator(self, actFunc)
        else:
            raise ValueError('pyNet::Neuron: activation function not supported.')

    def setBias(self, bias):
        self.setVar(bias)

    def getBias(self):
        return self.getVar()

    ## Network Traversion
    def feedForward(self, inDataList):

        if self.bias != None:
            inDataList.append(self.bias)

        self.potential  = self.inputOperator(inDataList)
        self.activation = self.activationFunc(self.potential)

    def backwardPropagation(self, inDataList):

        if self.bias != None:
            inDataList.append(self.bias) ####### WHY DO YOU APPEND THE BIAS? THE BIAS GRQDIENT IS THE GRADIENT AFTER THE ACTIVATION FUNCTION!?

        self.potential_  = self.inputOperatorGrad(inDataList)
        self.activation_ = self.activationFuncGrad(self.potential)

"""_________________________________________________________________________"""
"""_________________________________________________________________________"""
"""_________________________________________________________________________"""

class pyNeuron(pyObject):

    __neuronsCreated = 0        # This will keep a count of neurons created

    def __init__(self,
                 _id      = None,
                 _bias    = None,
                 _weights = None,
                 _operatorKey      = None,# input operator default = dot-product
                 _transferFunction = TF(1,1,dt=1.0),
                 _gradientFunction = None,
                 _sources  = [],
                 _sinks    = [],
                 _network  = None,
                 _type     = None,
                 _name     = None):
        '''
        Parameters:
        - _id: tuple, comprised of (layer id, node index in layer, network index)
        '''

        pyNeuron.__neuronsCreated += 1
        super(pyNeuron, self).__init__(_network=network, _type=self.__class__.__name__, _name=name)
        if _id != None:
            self._id = _id

        self.transferFunction = transferFunction # transfer function
        self.gradientFunction = gradientFunction # transfer function gradient
        self.bias     = bias
        self.weights  = self.weights
        self.output   = 0
        self.dendrites = {SIDE.IN: [], SIDE.OUT: []}
        if sources: self.sources  = self.dendrites[SIDE.IN]  = sources
        if sinks:   self.sinks    = self.dendrites[SIDE.OUT] = sinks

        self.nInputs  = len(self.sources)
        self.nOutputs = len(self.sinks)

        self.synapse = Synapse(self._id, _type, _name,
                               inputOperator=
                               None)
        self.synapse.connect(self.sources)

        self.soma    = Soma(   self._id, _type, _name,
                               transferFunction=None)

        self.axon    = Axon(   self._id, _type, _name,
                               inputOperator=None)
        self.axon.connect(self.sinks)

    def __str__(self):
        s = super(pyNeuron, self).__repr__()
        s += ' -> I am a {name}'.format(name=self.__class__.__name__)
        return s

    def __repr__(self):
        return "pyNeuron[{}]".format(self._id)

    def get_neuronsCreated(self):
        return self.__neuronsCreated

    ## Setters
    def setTransferFunction(self, transferFunction):
        if transferFunction in TRANSFER_FUNCTION:
            self.transferFunction = transferFunction
        else:
            raise ValueError('pyNet::pyNeuron: transfer function not supported.')

    def setBias(self, bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def setWeights(self, weights):
        self.weights = weights

    def getWeights(self):
        return self.weights

    def connect(self, port = SIDE.IN, neurons = []):
        if type(neurons) is list:
            for neuron in neurons:
                self.dendrites[port].append(neuron)
        elif type(neurons) is self._type:
            print(neurons)
            self.dendrites[port].append(neurons)

    ## Network Traversion
    def feedForward(self):

        if self.bias != None:
            inDataList.append(self.bias)

        self.potential  = self.inputOperator(inDataList)
        self.activation = self.activationFunc(self.potential)

    def backwardPropagation(self, inDataList):

        if self.bias != None:
            inDataList.append(self.bias) ####### WHY DO YOU APPEND THE BIAS? THE BIAS GRQDIENT IS THE GRADIENT AFTER THE ACTIVATION FUNCTION!?

        self.potential_  = self.inputOperatorGrad(inDataList)
        self.activation_ = self.activationFuncGrad(self.potential)

"""_________________________________________________________________________"""
"""_________________________________________________________________________"""
"""_________________________________________________________________________"""

'''----------------------------------------------------------------------
Example of pyAxon
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
def testpyAxon():
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
    testpyAxon()

# --- END OF FILE ---





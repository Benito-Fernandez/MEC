"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    from  pyObject            import pyObject
    from  pyProcessingElement import ProcessingElement
else:
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

class Soma(ProcessingElement):

    __somasCreated = 0        # This will keep a count of Somas created

    def __init__(self,
             _id   = None,
             _type = None,
             _name = None,
             _transferFunction = None, # transfer function, default = 1
             _neuron  = None,     # owning neuron
             _layer   = None,     # layer   that owns its neuron
             _ganglia = None,     # ganglia that owns its layer
             ):
        '''
        Parameters:
        - _id: tuple, comprised of (layer id, node index in layer, network index)
        '''

        Soma.__somasCreated += 1
        super(Soma, self).__init__(_id, bias, inputOperator, activationFunc)
        if _id != None:
            self._id = _id

        # Assigning to new variable names (keeps the same memory address)
#        if self.outputOperator in ACTIVATION_FUNCTION:
#            self.activationFunc     = self.outputOperator     # activation function
#            self.activationFuncGrad = self.outputOperatorGrad # activation function gradient
#        else:
#            raise ValueError('pyNet::Soma: activation function not supported.')
        self.activationFunc     = self.outputOperator     # activation function
        self.activationFuncGrad = self.outputOperatorGrad # activation function gradient
        self.bias       = self.variable
        self.potential  = self.state
        self.activation = self.output

    def __repr__(self):
        s = super(Soma, self).__repr__()
        s += ' -> I am a {name}'.format(name=self.__class__.__name__)
        return s

    def get_somasCreated(self):
        return self.__somasCreated

    ## Setters
    def setActivationFunc(self, actFunc):
        if actFunc in ACTIVATION_FUNCTION:
            self.setOutputOperator(self, actFunc)
        else:
            raise ValueError('pyNet::Soma: activation function not supported.')

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

class pySoma(pyObject):

    __somasCreated = 0        # This will keep a count of Somas created

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

        pySoma.__somasCreated += 1
        super(pySoma, self).__init__(_network=network, _type=self.__class__.__name__, _name=name)
        if _id != None:
            self._id = _id

        self.transferFunction = transferFunction # transfer function
        self.gradientFunction = gradientFunction # transfer function gradient
        self.bias     = bias
        self.weights  = self.weights
        self.output   = 0
        if sources: self.sources  = sources
        if sinks:   self.sinks    = sinks
        self.nInputs  = len(self.sources)
        self.nOutputs = len(self.sinks)

    def __repr__(self):
        s = super(pySoma, self).__repr__()
        s += ' -> I am a {name}'.format(name=self.__class__.__name__)
        return s

    def get_somasCreated(self):
        return self.__somasCreated

    ## Setters
    def setTransferFunction(self, transferFunction):
        if transferFunction in TRANSFER_FUNCTION:
            self.transferFunction = transferFunction
        else:
            raise ValueError('pyNet::pySoma: transfer function not supported.')

    def setBias(self, bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def setWeights(self, weights):
        self.weights = weights

    def getWeights(self):
        return self.weights

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




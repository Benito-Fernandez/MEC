from   networkx import DiGraph
import numpy        as np
"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    from  pyNeuron import Neuron
else:
    from .pyNeuron import Neuron
"""-------------------------------------------------------------------------"""

# Base Layer object
class Layer(DiGraph):

    __uid = 0		# This id will be maintained by the class and should not be edited by user.

    '''
    - Parameters:
		- layerPosition: Layer position in the neural network
		- nNodes: Optional argument, integer count of the nodes to be created in this layer instance
			- Passing no argument will create a layer without nodes.
			- If nodes are created, they will be created with the default value of the Node class.
		- type: Optional argument, type of layer (referring to how it will be connected)
			- Passing no argument will create a densely connected layer.
    '''
    def __init__(self, layerPosition = 0, nNodes = 0, networkId = None, layerType = 'dense', layerLabel = 'un-named_layer'):

        super(Layer, self).__init__()

        self.__uid = Layer.__uid
        Layer.__uid += 1
        self.label = layerLabel
        self.nNodes = nNodes
        self.layerPosition = layerPosition # may not be needed as position is handled by GNNet.layers = []
        self.type = layerType # {'dense', 'sparse'}

        if networkId == None:
            # If a networkId is not specified (i.e. this is just a layer not belonging to a network originally) then a modified layer id will be used.
            self.networkId = 'L[' + str(self.__uid) + ']'
        else:
#            self.networkId = 'N[' + str(networkId) + ']'
            self.networkId = networkId

        if self.nNodes > 0:
            for n in np.arange(nNodes):
                # neuronId = (layerPosition, n, networkId)
                neuronId = (self.__uid, n, self.networkId)
                self.add_node(neuronId, neuron = Neuron(_id = neuronId), neuronId = neuronId)
                # neuronId = neuronId attribute is added for drawing purposes

    def Id(self):
        return self.networkId

    def name(self):
        return self.label

    def __str__(self):
        s = 'Network[{}] Layer({}) with {} neurons. Label: {} - (UID:{})'
        return s.format(self.networkId,
                        self.layerPosition,
                        self.nNodes,
                        self.label,
                        self.__uid)

    def __repr__(self):
        s = 'I am a {name}'
        return s.format(name=self.__class__.__name__)

    '''
      - activationFunc: Activation function type from ActivationFunctionSet
    '''
    def setActivation(self, neuronId, activationFunc):
        self.nodes[neuronId]['neuron'].setActivation(activationFunc)

    def getActivation(self, neuronId=None):
        if neuronId is None:
            activations = []
            for node in self.nodes(data = True):
                activations.append(node['neuron'].activationFunc)
            return activations
        else:
            return self.nodes[neuronId]['neuron'].activationFunc

    def setAllActivation(self, activationFunc):
        for n, node in self.nodes(data = True):
            node['neuron'].setActivation(activationFunc)

    def setInputOperator(self, neuronId, inputOperator):
        self.nodes[neuronId]['neuron'].setInputOperator(inputOperator)

    def getInputOperator(self, neuronId):
        return self.nodes[neuronId]['neuron'].inputOperator

    def setAllInputOperator(self, inputOperator):
        for n, node in self.nodes(data = True):
            node['neuron'].setInputOperator(inputOperator)

    def setBias(self, neuronId, bias):
        self.nodes[neuronId]['neuron'].setBias(bias)

    def getBias(self, neuronId):
        return self.nodes[neuronId]['neuron'].bias

    def setAllBias(self, bias):
        for n, node in self.nodes(data = True):
            node['neuron'].setBias(bias)

    ### Layer data management
    def feedForward(self):

        print("Layer id: ", self.__uid, ", feedForward")

        for currNodeId in self.nodes:

            inDataList = []

        for preNodeId in self.predecessors(currNodeId):

            actvPrev = self.nodes[preNodeId]['neuron'].activation 	# activation of previous node
            axon = self[preNodeId][currNodeId]['axon']
            axon.feedForward(actvPrev)
            inDataList.append(axon.output)

            self.nodes[currNodeId]['neuron'].feedForward(inDataList)

    def feedForwardSetData(self, inData):
        '''
        inData is expected to be a 1D list of the following format:
	        [i1, i2, ..., ij]
	        For a data point containing j values (there must be j nodes in this layer)
        '''

        # TODO: [?] Does this check slow down feedforward much?

        nodeCount = len(self.nodes())
        dataCount = len(inData)

        if nodeCount != dataCount:
            print("Error: Number of nodes (", nodeCount, ") does not match numbe of data values (", dataCount, ").")

        print("Layer id: ", self.__uid, ", feedForwardSetData")
        print(inData)

        # [?] Is the first node always getting the first value (same for other nodes)?
        for index, currNodeId in enumerate(self.nodes):
            self.nodes[currNodeId]['neuron'].feedForward([ inData[index] ])

    def getNodesActivation(self):
        '''
		- Returns a list of output of every node
        '''

        activationList = []

        for neuronId in self.nodes:
            activationList.append( self.nodes[neuronId]['neuron'].activation )

        return activationList


        # def backwardPropagation(self, inData):

        # 	if self.bias != None:
        # 		inData.append(self.bias)

        # 	self.potential_ = self.inputOperatorGrad(inData)
        # 	self.activation_ = self.activationFuncGrad(self.potential)


















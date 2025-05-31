#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""
"""
@author: benito
         benito.fernandez@gmail.com
(c) 2018 SPAT, LLC
         Salute Physique Aesthetica Technolgie, LLC
         Austin, Texas, USA - The City of Ideas
"""
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""
#--------------------------------------------------
import matplotlib.pyplot  as plt
import networkx           as nx
from   networkx       import DiGraph
import numpy              as np
#--------------------------------------------------
if __name__ == "__main__":
    from   pyLayer         import Layer
    from   pyAxon          import Axon
    from   pyDefinitions   import WEIGHT_TYPE
else:
    from  .pyLayer         import Layer
    from  .pyAxon          import Axon
    from  .pyDefinitions   import WEIGHT_TYPE
#--------------------------------------------------
"""-------------------------------------------------------------------------"""


class Ganglia(DiGraph):

    _id = 0     # This id will be maintained by the class and should not be edited by user.

    ### Network Initialization and Setup

    def __init__(self, topology = []):
        '''
		Parameters
		- topology: List stating how many nodes there will be per layer.
             The index of the list is used as the index of the layer.
             The value at each index corresponds to the number of nodes
             in that layer.
        '''

        super(Ganglia, self).__init__()

        self._id = Ganglia._id
        Ganglia._id += 1

        # self.topology = topology
        self.layers = []

        # Adding Layers and nodes graph to Network graph
        # TODO: layerIndx may not be needed, see Layer class
        for layerIndx, nodeCount in enumerate(topology):
            newLayer = Layer(layerIndx, nodeCount, self._id)
            self.layers.append(newLayer)
            self.add_nodes_from(newLayer.nodes(data = True))
            # Layer is a graph itself. By using data = true, the data from the layer's graph persists to the network
        self.nLayers = len(self.layers)

    def __str__(self):
        return str(self._id)

    def __repr__(self):
        s = 'I am a {} with ({}) layers.'.format(self.__class__.__name__,
                                                 self.nLayers)
        return s

    def connectNetwork(self, connectionDict, weights = WEIGHT_TYPE['uniRand0To1'][0]):
        '''
        '''
        for fromId, toIds in connectionDict.items():
            for toId in toIds:
                axonId = (fromId, toId)
                self.add_edge(fromId, toId, axon = Axon(axonId, weights))

    ### Growing Network

    def preAppendNetworks(self, networks, connectionDict):
        '''
		- Parameters:
			- networks: list of neural network objects
			- connectionDict: dictionary that defines connections
				- see self.connectNetwork(...) method for expected dictionary format
        '''
        self._aggregateGraphs(networks, connectionDict, 0)

    def postAppendNetworks(self, networks, connectionDict):
        '''
		- Parameters:
			- networks: list of neural network objects
			- connectionDict: dictionary that defines connections
        '''
        self._aggregateGraphs(networks, connectionDict, 1)

    def preAppendLayers(self, layers, connectionDict):
        '''
		- Parameters:
			- networks: list of Layer objects
			- connectionDict: dictionary that defines connections
        '''
        self._aggregateGraphs(layers, connectionDict, 0, True)

    def postAppendLayers(self, layers, connectionDict):
        '''
		- Parameters:
			- networks: list of Layer objects
			- connectionDict: dictionary that defines connections
        '''
        self._aggregateGraphs(layers, connectionDict, 1, True)

    def _aggregateGraphs(self, graphs, connectionDict, beforeOrAfter, isLayersList = False):
        #### HOW ABOUT AGGREGATE/CONNECT NETWORKS? ONLY ConnectivityMatrix REQUIRED IS THE ConnectivityMatrrix
        #### FROM THE OUTPUT LAYER OF THE fromNetwork TO THE INPUT LAYER OF THE toNetwork???
        #### YOU SHOULD BE ABLE TO CREATE A
        #### ConnectedNetwork = ConnetNetworks (fromNetwork, toNetwork, ConnectivityMatrix)
        #### ConnectedNetwork = fromNetwork.ConnetNetwork (self, toNetwork, ConnectivityMatrix)
        #### That you can check if fromNetwork OR toNetwork are Layers (type) AND PERFORM THE
        #### CONNECTION TO THE Layer.
        ####
        #### THE ConnectivityMatrix SHOULD BE A Dictionary:
        #### {(fromLayer,fromNode): (toLayer,toNode)}
        #### WHERE: (fromLayer,fromNode) IS FROM THE fromNetwork AND
        ####        (toLayer,toNode)IS FROM THE toNetwork
        #### BOTH RELATIVE TO THE SPECIFIC NETWORK, NOT GLOBAL LAYERS NUMBERS
        #### THE NEW NETWORK WILL HAVE ITS OWN LAYERS THAT START AT 0 FOR THE INPUT LAYER!
        #### IT MAY OVERIDE THE fromNetwork IF IT IS A METHOD OF fromNetwork

        '''
		Private function
		- Parameters:
			- graphs: list of graph objects
			- connectionDict: dictionary that defines connection between any nodes in the network
			- beforeOrAfter: Bool
				- 0: layers appended before network
				- 1: layers appended after network
        '''

        # Error checking
        if not(beforeOrAfter == 0 or beforeOrAfter == 1):
            print("Error: Do not use _appendLayers method. It is a private method.")

        # Adding nodes and edges from each layer in layers, keeping data held in each
        for graph in graphs:
            self.add_nodes_from(graph.nodes(data = True))
            self.add_edges_from(graph.edges(data = True))

        # Appending layers before/after the current network's layers
        if isLayersList:
            if beforeOrAfter == 0:
                # append before network
                self.layers = graphs + self.layers
            else:
                # append after network
                self.layers = self.layers + graphs
        else:
            if beforeOrAfter == 0:
                # append before network
                for graph in reversed(graphs):
                    self.layers = graph.layers + self.layers
            else:
                # append after network
                for graph in graphs:
                    self.layers = self.layers + graph.layers

        # Connect new layers
        self.connectNetwork(connectionDict)


    ### Displaying Network

    def drawNetwork(self, withPositionLabel = False, withWeightLabel = False,
                    font_size=8, node_size=100, node_color='r', node_shape='o',
                    left=None, right=None, bottom=None, top=None, radius=None,
                    font_family='sans-serif'):
        fontSize,  nodeSize,  nodeColor,  nodeShape,  fontFamily = \
        font_size, node_size, node_color, node_shape, font_family

        if left   == None: left   = 0.0
        if right  == None: right  = 1.0
        if bottom == None: bottom = 0.0
        if top    == None: top    = 1.0
        maxNodes = 0
        for layer in self.layers:
            maxNodes = max(maxNodes,layer.nNodes)
        v_spacing = (top - bottom)/float(maxNodes)
        h_spacing = (right - left)/float(self.nLayers - 1)
        if radius == None: radius = v_spacing/5.
	    # TODO:
	    # - Clean up code.
	    # - Can this be moved to an external class (a drawing helper class)?

	    # TODO: Streamline this
	    # Adding "pos" attribute to each node
	    # 	- Note: This attribute is added every time this network is drawn because other networks or layers may have been appended to this network.
#        print(len(self.layers))
        for layerIndex, nodeList in enumerate(self.layers):
            layer_top = v_spacing*(len(nodeList) - 1)/2. + (top + bottom)/2.0
#            print(layerIndex,len(nodeList),self.layers)
            for nodeIndex, nodeId in enumerate(nodeList):
#                print(layerIndex,nodeIndex,nodeId)
                self.node[nodeId]['pos'] = (layerIndex*h_spacing + left, layer_top - nodeIndex*v_spacing) # (layerIndex, nodeIndex)
                # node_radius = v_spacing/4.

        pos = nx.get_node_attributes(self, 'pos')
#        print('pos',pos)
#        print('nodes',self.nodes)
        labelOffset = (0.3,0.1)
        lpos = {}
        #### networkx MAY HAVE AN OFFSET FOR THE LABEL AND JUSTIFICATION
        for p in pos:
            lpos[p]=(pos[p][0]+labelOffset[0],pos[p][1]+labelOffset[1])
        nx.draw(self, pos, node_size=nodeSize, node_color=nodeColor, \
                node_shape=nodeShape)
        # nx.draw_nx(self, pos, with_labels = False)
        node_labels = nx.get_node_attributes(self, 'somaId')
        # draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k',
        # font_family='sans-serif', font_weight='normal', alpha=1.0, ax=None, **kwds
        nx.draw_networkx_labels(self, lpos, node_labels, font_size=fontSize, \
                                font_family = fontFamily, label='pyGNNet', legend=True)
        # edge_labels = nx.get_edge_attributes(theGraph, 'value')
        # nx.draw_nx_edge_labels(self, pos, edge_labels=edge_labels)
#        plt.legend()
        plt.draw()
        plt.show()

	    # positions = nx.get_node_attributes(self, 'pos')

	    # if withPositionLabel:
	    # 	nx.draw(self, positions, with_labels = True)
	    # else:
	    # 	nx.draw(self, positions)

	    # if withWeightLabel:
	    # 	labels = nx.get_edge_attributes(self, 'axon')
	    # 	nx.draw_nx_edge_labels(self, positions, Nedge_labels = labels)

	    # plt.draw()
	    # plt.show()

    def connected_component_iter(self):
        """
        Yields connected components.
        """
        assert self.is_built is True
        for subgraph in nx.connected_component_subgraphs(self):
            yield subgraph

    def subgraph(self, nbunch):
        bunch = self.nbunch_iter(nbunch)
        # create new graph and copy subgraph into it
        H = self.__class__()
        # copy node and attribute dictionaries
        for n in bunch:
            H.node[n]=self.node[n]
        # namespace shortcuts for speed
        H_adj = H.adj
        self_adj=self.adj
        # add nodes and edges (undirected method)
        for n in H.node:
            Hnbrs={}
            H_adj[n]=Hnbrs
            for nbr,d in self_adj[n].items():
                if nbr in H_adj:
                    # add both representations of edge: n-nbr and nbr-n
                    Hnbrs[nbr]=d
                    H_adj[nbr][n]=d
        H.graph=self.graph
        return H

    ### Traversing Network
    def feedForward(self, dataSet):
        '''
	    dataSet is expected to be a 2D list of the following format:
	        [
	        [i1 (k = 1), i2 (k = 1), ..., ij (k = 1)]
	        [i1 (k = 2), i2 (k = 2), ..., ij (k = 2)]
	        .
	        .
	        .
	        [i1 (k = K), i2 (k = K), ..., ij (k = K)]
	        ]
	        For a data set containing k points, and j input nodes (j inputs must match network's topology)
        '''
	    # TODO: set dataSet to input nodes to

	    # tempLayers = self.layers     # tempLayers will be a modified version of layers
	    # firstLayer = tempLayers.pop(0)
	    # lastLayer = tempLayers.pop()

        layersCount = len(self.layers)
        transversableLayersRange = np.arange(1, layersCount) # every layer except input layer

        for point in dataSet:

            # Input layer:
            self.layers[0].feedForwardSetData(point)

            # Transversable layers:
        for layerIndx in transversableLayersRange:
            self.layers[layerIndx].feedForward()

            ## --- Debug print
            networkOut = self.layers[-1].getNodesActivation()
            print("Network Output: ", networkOut)
            ## ---

            # Output layer activation:
            networkOut = self.layers[-1].getNodesActivation()

            print("Network Output: ", networkOut)


    # def logCurrentSetup(self):

    # 	currentLayer = -1

    # 	print("\nSuccessors Somas With Corresponding Axon Per Parent Soma")

    # 	for layerIndex, nodeCount in enumerate(self.topology):
    # 		for nodeIndx in np.arange(nodeCount):

    # 		    # Prints layer
    # 			if currentLayer != layerIndex:
    # 				currentLayer = layerIndex
    # 				print("------------------------")
    # 				print("------------------------")
    # 				print("Layer: ", currentLayer, "\n")

    # 			currNodeId = (layerIndex, nodeIndx)

    # 			title = "(" + str(currNodeId[0]) + ", " + str(currNodeId[1]) + ")"

    # 			tableData = [["Soma ID", "Soma In. Op.", "Soma Act.", "Axon ID", "Axon Weight", "Axon In. Op.", "Axon Out. Op."]]

    # 			for succNodeId in self.successors(currNodeId):

    # 				axon = self[currNodeId][succNodeId]['axon']
    # 				succSoma = self.nodes[succNodeId]['soma']

    # 				somaInOp = succSoma.inputOperatorDict[2] if succSoma.inputOperatorDict != None else None
    # 				somaAct = succSoma.outputOperatorDict[2] if succSoma.outputOperatorDict != None else None
    # 				axonInOp = axon.inputOperatorDict[2] if axon.inputOperatorDict != None else None
    # 				axonOutOp = axon.outputOperatorDict[2] if axon.outputOperatorDict != None else None


    # 				tableData.append([succNodeId, somaInOp, somaAct, axon._id, axon.weight, axonInOp, axonOutOp])

    # 			tableInstance = SingleTable(tableData, title)
    # 			print(tableInstance.table)
    # 			print()

    # def logMemberValues(self):

    # 	currentLayer = -1

    # 	print("\nSuccessors Somas With Corresponding Axon Per Parent Soma")

    # 	for layerIndex, nodeCount in enumerate(self.topology):
    # 		for nodeIndx in np.arange(nodeCount):

    # 		    # Prints layer
    # 			if currentLayer != layerIndex:
    # 				currentLayer = layerIndex
    # 				print("------------------------")
    # 				print("------------------------")
    # 				print("Layer: ", currentLayer, "\n")

    # 			currNodeId = (layerIndex, nodeIndx)
    # 			currSoma = self.nodes[succNodeId]['soma']

    # 			title = "(" + str(currNodeId[0]) + ", " + str(currNodeId[1]) + ") pot: " + currSoma.potential + " act: " + currSoma.activation

    # 			tableData = [["Soma ID", "Soma Pot.", "Soma Act.", "Axon ID", "Axon State", "Axon Out."]]

    # 			for succNodeId in self.successors(currNodeId):

    # 				axon = self[currNodeId][succNodeId]['axon']
    # 				succSoma = self.nodes[succNodeId]['soma']

    # 				tableData.append([succNodeId, succSoma.potential, succSoma.activation, axon._id, axon.state, axon.output])

    # 			tableInstance = SingleTable(tableData, title)
    # 			print(tableInstance.table)
    # 			print()



    # old feedforward
	    # tplg = self.topology    # local copy of topology
	    # frstLyrNC = tplg.pop(0) # first layer node count
	    # frstLyrIdx = 0    # first layer index

	    # for dataPoint in dataSet:

	    # 	print(dataPoint)

	    # 	for nodeIndx in np.arange(frstLyrNC):
	    # 		somaId = (frstLyrIdx, nodeIndx)
	    # 		print(somaId, dataPoint[nodeIndx])

	    # 		self.nodes[somaId]['soma'].feedForward(dataPoint[nodeIndx])
	    # 		print(self.nodes[somaId]['soma'].activation)

	    # 	for layerIndex, nodeCount in enumerate(tplg, start = 1):

	    # 		for nodeIndx in np.arange(nodeCount):

	    # 			currNodeId = (layerIndex, nodeIndx)

	    # 			inData = []

	    # 			for preNodeId in self.predecessors(currNodeId):

	    # 				actvPrev = self.nodes[preNodeId]['soma'].activation     # activation of previous node
	    # 				print(actvPrev)
	    # 				self[preNodeId][currNodeId]['axon'].feedForward(actvPrev)
	    # 				inData.append(actvPrev * weight)

	    # 			self.nodes[currNodeId]['soma'].feedForward(inData)

	    # self.logMemberValues()




    # def backPropagation(self):

    #     # TODO: HOW TO ADD LOSS GRADIENT?
    # 	for pair in reversed(self.topology):
    # 		for nodeIndx in np.flip(np.arange(pair[1]), 0):

    # 		    # currNodeId = (pair[0], nodeIndx)
    # 		    # node = self.nodes[currNodeId]['soma']

    # 		    # inData = []

    # 		    # for preNodeId in self.successors(currNodeId):
    # 		    # 	actvPrev = self.nodes[preNodeId]['soma'].activation     # activation of previous node
    # 		    # 	weight = self[preNodeId][currNodeId]['weight'].weight


    # 		    # 	inData.append(actvPrev * weight)

    # 		    # node.backPropagation(inData)


if __name__ == '__main__':
    (width, height) = (10, 10)
    (left, right, bottom, top) = (.1, .9, .1, .9) # full = (0.0, 1.0, 1.0, 0.0); fraction = (.1, .9, .1, .9)
    layer_sizes = [2, 7, 3]

    topology = [1, 3, 2]
    network = Ganglia(topology)
    connectionDict = {
    	(4, 0, 1): [(5, 0, 1), (5, 1, 1), (5, 2, 1), (6, 1, 1)],
    	(5, 0, 1): [(6, 0, 1)],
    	(5, 1, 1): [(5, 0, 1), (6, 0, 1)],
    	(5, 2, 1): [(6, 0, 1), (6, 1, 1)],
    	(6, 1, 1): [(5, 1, 1), ]
    	}
    network.connectNetwork(connectionDict)

    fig = plt.figure(2,figsize=(width, height)) # (width, height) [inches]
    fig.clf()
    ax = fig.gca()
    network.drawNetwork(node_color='g', node_shape='s', font_family='Candara')
    ax.axis('equal')
    fig.draw

"""
def draw_networkx(G, pos=None, arrows=True, with_labels=True, **kwds):
    '''Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See :py:mod:`networkx.drawing.layout` for functions that
       compute node positions.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
        For directed graphs, choose the style of the arrowsheads.
        See :py:class: `matplotlib.patches.ArrowStyle` for more
        options.

                The following classes are defined

                Class	Name	Attrs
                Curve	-	None
                CurveB	->	head_length=0.4,head_width=0.2
                BracketB	-[	widthB=1.0,lengthB=0.2,angleB=None
                CurveFilledB	-|>	head_length=0.4,head_width=0.2
                CurveA	<-	head_length=0.4,head_width=0.2
                CurveAB	<->	head_length=0.4,head_width=0.2
                CurveFilledA	<|-	head_length=0.4,head_width=0.2
                CurveFilledAB	<|-|>	head_length=0.4,head_width=0.2
                BracketA	]-	widthA=1.0,lengthA=0.2,angleA=None
                BracketAB	]-[	widthA=1.0,lengthA=0.2,angleA=None,widthB=1.0,lengthB=0.2,angleB=None
                Fancy	fancy	head_length=0.4,head_width=0.4,tail_width=0.4
                Simple	simple	head_length=0.5,head_width=0.5,tail_width=0.2
                Wedge	wedge	tail_width=0.3,shrink_factor=0.5
                BarAB	|-|	widthA=1.0,angleA=None,widthB=1.0,angleB=None

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    with_labels :  bool, optional (default=True)
       Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional (default G.nodes())
       Draw only specified nodes

    edgelist : list, optional (default=G.edges())
       Draw only specified edges

    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    node_color : color string, or array of floats, (default='r')
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_shape :  string, optional (default='o')
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8'.

                marker	description
                "."	point
                ","	pixel
                "o"	circle
                "v"	triangle_down
                "^"	triangle_up
                "<"	triangle_left
                ">"	triangle_right
                "1"	tri_down
                "2"	tri_up
                "3"	tri_left
                "4"	tri_right
                "8"	octagon
                "s"	square
                "p"	pentagon
                "P"	plus (filled)
                "*"	star
                "h"	hexagon1
                "H"	hexagon2
                "+"	plus
                "x"	x
                "X"	x (filled)
                "D"	diamond
                "d"	thin_diamond
                "|"	vline
                "_"	hline
                TICKLEFT	tickleft
                TICKRIGHT	tickright
                TICKUP	tickup
                TICKDOWN	tickdown
                CARETLEFT	caretleft (centered at tip)
                CARETRIGHT	caretright (centered at tip)
                CARETUP	caretup (centered at tip)
                CARETDOWN	caretdown (centered at tip)
                CARETLEFTBASE	caretleft (centered at base)
                CARETRIGHTBASE	caretright (centered at base)
                CARETUPBASE	caretup (centered at base)
                "None", " " or ""	nothing

    alpha : float, optional (default=1.0)
       The node and edge transparency

    cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of nodes

    vmin,vmax : float, optional (default=None)
       Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    width : float, optional (default=1.0)
       Line width of edges

    edge_color : color string, or array of floats (default='r')
       Edge color. Can be a single color format string,
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional (default=None)
       Minimum and maximum for edge colormap scaling

    style : string, optional (default='solid')
       Edge line style (solid|dashed|dotted,dashdot)

    labels : dictionary, optional (default=None)
       Node labels in a dictionary keyed by node of text labels

    font_size : int, optional (default=12)
       Font size for text labels

    font_color : string, optional (default='k' black)
       Font color string

    font_weight : string, optional (default='normal')
       Font weight

    font_family : string, optional (default='sans-serif')
       Font family

           avail_font_names = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
            ['STIXNonUnicode', 'DejaVu Serif', 'DejaVu Sans Mono', 'DejaVu Sans',
             'cmsy10', 'DejaVu Serif', 'STIXGeneral', 'cmss10', 'DejaVu Sans',
             'DejaVu Sans Mono', 'STIXSizeThreeSym', 'DejaVu Sans',
             'STIXSizeFourSym', 'STIXSizeTwoSym', 'DejaVu Sans Mono',
             'DejaVu Serif Display', 'STIXSizeOneSym', 'STIXSizeOneSym',
             'DejaVu Serif', 'STIXNonUnicode', 'STIXNonUnicode', 'DejaVu Sans Mono',
             'STIXSizeFourSym', 'cmmi10', 'STIXGeneral', 'STIXNonUnicode',
             'STIXSizeThreeSym', 'cmex10', 'cmr10',
             'STIXGeneral',
             'STIXSizeTwoSym',
             'cmtt10',
             'cmb10',
             'DejaVu Sans Display',
             'STIXGeneral',
             'DejaVu Serif',
             'DejaVu Sans',
             'STIXSizeFiveSym',
             'Trebuchet MS',
             'Palatino Linotype',
             'Gabriola',
             'Comic Sans MS',
             'Constantia',
             'Mongolian Baiti',
             'Wingdings',
             'Ebrima',
             'Segoe UI',
             'Courier New',
             'Cambria',
             'Consolas',
             'Microsoft Himalaya',
             'Microsoft Tai Le',
             'Candara',
             'Corbel',
             'Nirmala UI',
             'Segoe Print',
             'Corbel',
             'Cambria',
             'Calibri',
             'Arial',
             'Courier New',
             'Segoe UI',
             'Segoe Script',
             'Verdana',
             'MV Boli',
             'Malgun Gothic',
             'Microsoft Tai Le',
             'Comic Sans MS',
             'Nirmala UI',
             'Tahoma',
             'Candara',
             'Segoe UI',
             'Trebuchet MS',
             'Constantia',
             'Comic Sans MS',
             'Segoe MDL2 Assets',
             'Times New Roman',
             'Microsoft New Tai Lue',
             'Cambria',
             'Georgia',
             'Palatino Linotype',
             'Verdana',
             'Segoe UI',
             'Segoe UI Symbol',
             'Segoe UI Historic',
             'Nirmala UI',
             'Candara',
             'Bahnschrift',
             'Nirmala UI',
             'Gadugi',
             'Calibri',
             'Constantia',
             'Leelawadee UI',
             'Microsoft New Tai Lue',
             'Myanmar Text',
             'Consolas',
             'Segoe UI',
             'Myanmar Text',
             'Corbel',
             'Lucida Sans Unicode',
             'Corbel',
             'Palatino Linotype',
             'SimSun-ExtB',
             'Franklin Gothic Medium',
             'Nirmala UI',
             'Microsoft PhagsPa',
             'Trebuchet MS',
             'Candara',
             'Sylfaen',
             'Franklin Gothic Medium',
             'Consolas',
             'Segoe Print',
             'Calibri',
             'Candara',
             'Malgun Gothic',
             'Segoe UI',
             'Georgia',
             'Arial',
             'Segoe UI',
             'Times New Roman',
             'Calibri',
             'Georgia',
             'Leelawadee UI',
             'Leelawadee UI',
             'Ink Free',
             'HoloLens MDL2 Assets',
             'Ebrima',
             'Arial',
             'Tahoma',
             'Arial',
             'Candara',
             'Constantia',
             'Segoe UI',
             'Javanese Text',
             'Trebuchet MS',
             'Webdings',
             'Calibri',
             'Segoe UI',
             'Segoe UI',
             'Verdana',
             'Lucida Console',
             'Calibri',
             'Gadugi',
             'Georgia',
             'Times New Roman',
             'Ink Free',
             'Nirmala UI',
             'Arial',
             'Leelawadee UI',
             'Segoe UI Emoji',
             'Segoe UI',
             'Candara',
             'Segoe Script',
             'Leelawadee UI',
             'Verdana',
             'Impact',
             'Times New Roman',
             'Malgun Gothic',
             'Courier New',
             'Marlett',
             'Microsoft PhagsPa',
             'Microsoft Yi Baiti',
             'Segoe UI',
             'Symbol',
             'Candara',
             'Comic Sans MS',
             'Consolas',
             'Palatino Linotype',
             'Microsoft Sans Serif',
             'Gabriola',
             'Courier New',
             'Leelawadee UI']
    label : string, optional
        Label for graph legend

    Notes
    -----
    For directed graphs, arrows  are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    >>> import matplotlib.pyplot as plt
    >>> limits = plt.axis('off')  # turn of axis

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html
    '''

"""





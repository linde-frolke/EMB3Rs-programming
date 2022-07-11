import numpy as np
import random
import pandas as pd

    # Create Network
def Create_Network(agent_ids,buildingID,nodes_name_data,pipe_length):
	
	# Form Network Graph Data
	buildingNames = [id[0] for id in buildingID.values]

	# Define the paper nodes
	paper = []
	nodes_name_dict = dict(zip((nodes_name_data.id),nodes_name_data.name))
	for node_ in nodes_name_dict.keys():
		paper.append(nodes_name_dict[node_])

	random.seed(42)

	# dictionary of location of agents in different nodes
	loc_dict = {}
	node_sm = [3] # define where the supermarket is located
	node_grid = [42] # define where the grid is located
	used_node = np.concatenate([node_sm,node_grid,buildingNames])
	rest_nodes = list(np.setdiff1d(paper,used_node)) # nodes without any agents (blue nodes)
	buildingIds = buildingNames.copy()

	# reset the name of the loc_dict
	for node in (paper):
		loc_dict[node] = f'node_{node}'

	# rename the nodes into agent names
	for i,agent in enumerate(agent_ids):
		if 'grid' in agent:
			loc_dict[node_grid.pop(0)] = agent
			continue
			
		if 'sm' in agent:
			loc_dict[node_sm.pop(0)] = agent
			continue
		
		# put the consumers into end nodes
		if buildingIds:
			loc_dict[buildingIds.pop(0)] = agent
			continue
		
		# put the rest of consumers in random blue nodes
		if rest_nodes[0] == 1 or rest_nodes[0]==2:
			loc_dict[rest_nodes.pop(0)] = agent
		else:
			node = rest_nodes.pop(random.randrange(len(rest_nodes)))
			loc_dict[node] = agent

	# Define pipe lengths and nodes
	pool_pipe_length = pd.DataFrame(pipe_length['Pipe_length'])
	from_node = []
	to_node = []
	for i,row in pipe_length.iterrows():
		from_node.append(loc_dict[int(row['up_str_node'])])
		to_node.append(loc_dict[int(row['dw_str_node'])])

	pool_pipe_length['up_str_node'] = from_node
	pool_pipe_length['dw_str_node'] = to_node

	# Nodes
	nodes = list(loc_dict.values())

	# Edges
	edges = [(up_,dw_) for up_,dw_ in pool_pipe_length.loc[:,'up_str_node':'dw_str_node'].values]
	
	return nodes, edges, pool_pipe_length

def Create_penalties(nodes, edges):

	class Graph():
		def __init__(self, vertices):
			self.V = vertices
			self.graph = [[0 for column in range(vertices)]
						for row in range(vertices)]
			self.weights = [0 for row in range(vertices)]
	
		def printSolution(self, dist, source_node):
			print(f"Vertex \t Distance from n{source_node+1}")
			for node in range(self.V):
				print(f"n{node+1}", "\t\t", dist[node])
	
		# A utility function to find the vertex with
		# minimum distance value, from the set of vertices
		# not yet included in shortest path tree
		def minDistance(self, dist, sptSet):
	
			# Initialize minimum distance for next node
			min = 1e7
	
			# Search not nearest vertex not in the
			# shortest path tree
			for v in range(self.V):
				if dist[v] < min and sptSet[v] == False:
					min = dist[v]
					min_index = v
	
			return min_index
	
		# Function that implements Dijkstra's single source
		# shortest path algorithm for a graph represented
		# using adjacency matrix representation
		def dijkstra(self, src):
	
			dist = [1e7] * self.V
			dist[src] = 0
			sptSet = [False] * self.V
	
			for cout in range(self.V):
	
				# Pick the minimum distance vertex from
				# the set of vertices not yet processed.
				# u is always equal to src in first iteration
				u = self.minDistance(dist, sptSet)
	
				# Put the minimum distance vertex in the
				# shortest path tree
				sptSet[u] = True
	
				# Update dist value of the adjacent vertices
				# of the picked vertex only if the current
				# distance is greater than new distance and
				# the vertex in not in the shortest path tree
				for v in range(self.V):
					if (self.graph[u][v] > 0 and
					sptSet[v] == False and
					dist[v] > dist[u] + self.graph[u][v]):
						dist[v] = dist[u] + self.graph[u][v]
						self.weights[v] = dist[v]
	
			#self.printSolution(dist,src)  # to see what edges the weights are associated to
			return (self.weights), self.graph
	
	# Make matrix of node connections
	undirected = False

	edges=list(edges)
	n_edges = len(edges)
	n_nodes = len(nodes)
	A = np.zeros((n_nodes,n_nodes))

	for i, node1 in enumerate((nodes)):
		for j,node2 in enumerate((nodes)):
			if (node1,node2) in edges:
				A[i,j] = 1
				if undirected == True:
					A[j,i] = 1 # if the graph is undirected
	
	source = 42 # Define source node (grid node)
	source_index = source - 1 
	g = Graph(len(nodes)) # Create graph 

	g.graph = A 
	g.graph.shape

	weights,graph=g.dijkstra(source_index)

	return weights # index of where the node is located
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
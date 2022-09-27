import numpy as np
import pandas as pd
import random


#def make_gis_data(agent_ids):
path = "market_module/DTU_case_study_new/Linde_data/"
nodes_name_data = pd.read_csv(path + 'Nodes_data.csv',names=['id','name'])
buildingID = pd.read_csv(path+'BuildingID.csv',names=['Node'])   
pipe_length = pd.read_csv(path+'pipe_data.csv')
pipe_dir = pd.read_csv(path+'Pipe_edges.csv',names=['from','to'])

# Form Network Graph Data
buildingNames = [id[0] for id in buildingID.values]
# Define the paper nodes
nodes = nodes_name_data.name.to_list()
nodes_name_dict = dict(zip((nodes_name_data.id),nodes_name_data.name))

random.seed(42)
# dictionary of location of agents in different nodes
loc_dict = {}
node_sm = [3]
node_grid = [42]
used_node = np.concatenate([node_sm,node_grid,buildingNames])
rest_nodes = list(np.setdiff1d(nodes,used_node))
buildingIds = buildingNames.copy()

# reset the name of the loc_dict
for node in (nodes):
    loc_dict[node] = f'node_{node}'

# rename the nodes into agent names
for i,agent in enumerate(agent_ids):
    if 'grid' in agent:
        loc_dict[node_grid.pop(0)] = agent
        continue
        
    if 'sm' in agent:
        loc_dict[node_sm.pop(0)] = agent
        continue
    
    # put the rest of the consumers into end nodes
    if buildingIds:
        loc_dict[buildingIds.pop(0)] = agent
        continue
    
    if rest_nodes[0] == 1 or rest_nodes[0]==2:
        loc_dict[rest_nodes.pop(0)] = agent
    else:
        node = rest_nodes.pop(random.randrange(len(rest_nodes)))
        loc_dict[node] = agent

# Define pipe lengths and nodes
#nodes_name_data, buildingID, pipe_length, pipe_dir = load_network()

pool_pipe_length = pd.DataFrame(pipe_length['Pipe_length'])
from_node = []
from_num = []
to_node = []
to_num = []

for i,row in pipe_length.iterrows():
    from_num.append(int(row['up_str_node']))
    to_num.append(int(row['dw_str_node']))
    from_node.append(loc_dict[int(row['up_str_node'])])
    to_node.append(loc_dict[int(row['dw_str_node'])])

pool_pipe_length['up_str_node'] = from_node
pool_pipe_length['up_str_num'] = from_num
pool_pipe_length['dw_str_node'] = to_node
pool_pipe_length['dw_str_num'] = to_num

#from_to
pipes = [(from_,to_) for from_,to_ in zip(from_num,to_num)]
from_to = [(loc_dict[from_],loc_dict[to_]) for from_,to_ in zip(from_num,to_num)]

#length
length_dict = {}

for from_,to_ in zip(from_node,to_node):
    length_dict[f'({from_},{to_})'] = pool_pipe_length[(pool_pipe_length['up_str_node'] == from_) 
    & (pool_pipe_length['dw_str_node'] == to_)]['Pipe_length'].values[0]

lengths = list(length_dict.values())

# losss_total 
losses_total = np.ones(len(from_to))



# make connectivity matrix
A = np.zeros((len(nodes), len(pipes)))
for p_nr in range(len(pipes)):
    p = pipes[p_nr]
    n1_nr = nodes.index(p[0])
    n2_nr = nodes.index(p[1])
    A[n1_nr, p_nr] = 1
    A[n2_nr, p_nr] = -1

# compute distance between any 2
distance = np.zeros((len(nodes), len(nodes)))
for i in nodes[:-1]:
    for j in nodes[:-1]:
        if not i == j:
            print(str(i) + " and " + str(j))
            B = np.zeros(len(nodes) -1)
            B[nodes.index(i)] = 1
            B[nodes.index(j)] = -1
            sol = np.linalg.solve(A[0:(A.shape[0]-1)], B ) 
            # select needed pipes
            sol = [bool(abs(x)) for x in sol]
            # add lengths
            distance[nodes.index(i),nodes.index(j)] = sum(np.array(lengths)[sol])

distance[41,:-1] = distance[40,:-1] + lengths[0]
distance[:-1, 41] = distance[41,:-1]


## Now convert this to the gis_data format 
from itertools import product
agents_from_to = [(i, j) for i,j in product(agent_ids, agent_ids) if i >= j]
distance_vec = [distance[list(loc_dict.keys())[list(loc_dict.values()).index(x[0])] -1 , list(loc_dict.keys())[list(loc_dict.values()).index(x[1])] - 1] for x in agents_from_to]
losses_vec =  distance_vec ## TODO make this one

# total_costs
total_costs = list(np.ones(len(agents_from_to)))

gis_data = {}
gis_data['from_to'] = agents_from_to
gis_data['losses_total'] = losses_vec
gis_data['length'] = distance_vec
gis_data['total_costs'] = total_costs

gis_data

pd.DataFrame(gis_data)

import json
gis_data
with open('Nordhavn_gis_data_for_sergio.json', 'w') as f:
    json.dump(gis_data, f)

nodes
with open('Nordhavn_nodes_data_for_sergio.json', 'w') as f:
    json.dump(nodes, f)

pipes
with open('Nordhavn_pipes_data_for_sergio.json', 'w') as f:
    json.dump(pipes, f)
    
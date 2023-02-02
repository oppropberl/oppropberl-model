# this is a rough version, use the ipynb version to get the final version

import glob, subprocess
from tqdm import tqdm
import re

# list_of_java_files = []

# for f in glob.glob('/home/anonymous/DLAnnot/cfg_out/*.txt', recursive=True):
#     list_of_java_files.append(f)

# for considering_file in tqdm(list_of_java_files):
#     print(considering_file)

#     file_name = considering_file.split("/")[-1][:-4]
#     print(file_name)

#     reading_file = open(considering_file, "r")
#     original_file = reading_file.read()

considering_file = "/home/anonymous/DLAnnot/cfg_out/DivideByZero1_y.txt"
file_name = considering_file.split("/")[-1][:-4]

print(considering_file)
print(file_name)

reading_file = open(considering_file, "r")

all_graphs = []
current_graph = ""
last_line_arrow = False
arrow_flag = False
for line in reading_file:
    if "->" in line and not arrow_flag: # first line
        current_graph += line
        last_line_arrow = True

    elif "->" not in line and not arrow_flag and last_line_arrow: # edges finished rn
        current_graph += line
        last_line_arrow = False
        arrow_flag = True

    elif "->" not in line and arrow_flag and not last_line_arrow: # edges finished already
        current_graph += line
        last_line_arrow = False
        arrow_flag = True

    elif "->" in line and arrow_flag: # start of a new graph
        all_graphs.append(current_graph)
        current_graph = line
        arrow_flag = False
        last_line_arrow = True

all_graphs.append(current_graph) # for the last graph

print(len(all_graphs))
print(all_graphs[0])

all_current_graph_edges = []
all_current_graph_nodes = []

for graph in all_graphs:
    graph = graph.split("\n")
    current_graph_edges = []
    current_graph_nodes = {}
    current_node = None
    for line in graph:

        line = line.strip()

        if ' -> ' in line:
            current_graph_edges.append(tuple(line.split(" -> ")))

        elif re.match(r"(\d+:)", line) is not None and current_node is not None: # new node started
            line = line[:-1] # ignore colon :
            current_graph_nodes[line] = ""
            current_node = line
        
        elif re.match(r"(\d+:)", line) is not None: # first time node
            line = line[:-1] # ignore colon :
            current_graph_nodes[line] = "" 
            current_node = line

        elif current_node is not None: # inside node 
            current_graph_nodes[current_node] += line
            current_graph_nodes[current_node] += "\n"
        
    all_current_graph_edges.append(current_graph_edges)
    all_current_graph_nodes.append(current_graph_nodes)

# copy to onedrive + run all + check one + add to csv + wandb upload + PyG try
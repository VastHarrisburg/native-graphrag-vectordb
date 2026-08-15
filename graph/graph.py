import networkx as nx
from processing_data import Processing_data

graph = Processing_data();
graph.process_nodes("data/nodes.json");
graph.process_edges("data/edges.json");



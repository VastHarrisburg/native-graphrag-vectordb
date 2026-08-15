import networkx as nx;
from pathlib import Path;
import re
class Graph_processing:
    #Add data structure to prevent duplicate nodes/edges from being created, make sure it stores multiple documents
    #In addition, need to develop a method for traversal to determine relatedness incase edge has multiple contexts
    def __init__(self):
        self.G = nx.MultiDiGraph();

    def add_chunk(self, node_type, document, content, name):
        normalized = self.normalize_node_id(name);
        if normalized not in self.G:
            self.G.add_node(name, type=node_type, document=document, content=content)

    def add_Origin(self, node_type, document, name):
        normalized = self.normalize_node_id(name);
        if normalized not in self.G:
            self.G.add_node(name, type=node_type, document=document)

    def find_related(self, node_id, depth):
        bfs_tree = nx.bfs_tree(self.G, node_id, depth_limit=2);
        subgraph = self.G.subgraph(related_nodes)
        nodes = list(subgraph.nodes())
        edges = list(subgraph.edges())
        return nodes, edges;
        
    def add_edge(self, node1, node2, context):
        self.G.add_edge(node1, node2, edge_text=context);
    
    def normalize_node_id(self, name):
        normalized = name.strip().lower()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def graph_return(self, node_id, depth):
        bfs_tree = nx.bfs_tree(self.G);
        subgraph = self.G.subgraph(related_nodes)
        nodes = list(subgraph.nodes())
        edges = list(subgraph.edges())
        return nodes, edges;
    

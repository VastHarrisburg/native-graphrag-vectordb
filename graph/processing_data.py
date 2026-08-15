import networkx as nx;
from pathlib import Path;
import json;
from graph_processing import Graph_processing
class Processing_data:
    
    def __init__(self):
        self.G = Graph_processing();

    def read_nodes(self, filePath):
        try:
            with Path(filePath).open(encoding="UTF-8") as source:
                data = json.load(source)
        except FileNotFoundError as error:
            print("File not found")
        except PermissionError as error:
            print("Permission denied:", error)
        except json.JSONDecodeError as error:
            print("Cannot decode json", error)
        return data;

    def process_nodes(self, filePath):
        path = filePath
        data = self.read_nodes(path)
        length = len(data)
        for i in range(0,length):
            if(data[i]["type"] == "chunk"):
                node_type = data[i]["type"]
                document = data[i]["document"]
                content = data[i]["content"]
                name = data[i]["name"]
                self.G.add_chunk(node_type, document, content, name)
            if(data[i]["type"] == "entity"):
                node_type = data[i]["type"]
                document = data[i]["document"]
                name = data[i]["name"]
                self.G.add_Origin(node_type, document, name)
    
    def read_edges(self, filePath):
        try:
            with Path(filePath).open(encoding="UTF-8") as source:
                data = json.load(source)
        except FileNotFoundError as error:
            print("File not found")
        except PermissionError as error:
            print("Permission denied:", error)
        except json.JSONDecodeError as error:
            print("Cannot decode json", error)
        return data;

    def process_edges(self, filePath):
        data = self.read_edges(filePath)
        length = len(data)
        for i in range(0,length):
            start = data[i]["source"]
            end = data[i]["target"]
            context = data[i]["context"]
            self.G.add_edge(start, end, context)
    
    def return_graph(self):
        self.G.graph_return();
    

        


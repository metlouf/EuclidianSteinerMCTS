import numpy as np
import networkx as nx
from scipy.spatial import distance
from itertools import product,combinations


class EuclideanSteinerTree:

    def __init__(self, points: np.ndarray):
        """
        Initialize the Steiner Tree with terminal points.
        :param points: A numpy array of shape (N, 2) or (N, 3) representing terminal points in 2D or 3D.
        """
        self.dim = points.shape[1]  # Determine if 2D or 3D
        self.max_degree = self.dim + 1

        self.terminals = set(map(tuple, points))  # Convert to set for quick lookup
        self.graph = nx.Graph()
        
        # Add terminal nodes to the graph
        for point in self.terminals:
            self.graph.add_node(point, type='terminal')
        
        self.component = [] # Start with all terminals unconnected
        self.track_connected = {}
        for node in self.graph.nodes :
            comp = {node}
            self.component.append(comp)
            self.track_connected[node] = comp
        

    def get_low_degree_terminals(self):
        """
        Return terminal points with 2 or fewer neighbors.
        """
        return {node for node in self.terminals
                 if self.graph.degree[node] <= self.max_degree}
    
    def legal_moves(self):
        moves = []
        for i in range(2,self.max_degree+1):
            moves+=self.get_combination(size = i)
        return moves
    
    def get_combination(self,size):
        moves = []
        mixes = combinations(self.component,size)
        for mix in mixes :
            for combo in product(*mix):
                moves.append(combo)
        return moves
    
    def terminal(self):
        return len(self.component)==1
    
    def get_hash(self):
        h=0
        return h
    
if __name__ == "__main__":
    # Define a simple test case: a square with a central Steiner point
    terminals = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    tree = EuclideanSteinerTree(terminals)
    
    print("Initial legal moves:")
    for move in tree.legal_moves():
        print(move)
    
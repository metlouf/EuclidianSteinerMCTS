import numpy as np
import networkx as nx
from scipy.spatial import distance
from itertools import product,combinations
import matplotlib.pyplot as plt

class EuclideanSteinerTree:

    def __init__(self, points: np.ndarray):
        """
        Initialize the Steiner Tree with terminal points.
        :param points: A numpy array of shape (N, 2) or (N, 3) representing terminal points in 2D or 3D.
        """
        self.dim = points.shape[1]  # Determine if 2D or 3D 
        #(the code now works for 2D only may update legal moves meth for 3D)
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
                 if self.graph.degree[node] <= self.dim}
    
    def legal_moves(self):
        moves = []
        ## add simple edges
        moves+=self.get_combination(size = 2)
        ## add edges with steiner point
        moves+=self.get_bifurcation()
        return moves
    
    def get_combination(self,size):
        moves = []
        mixes = combinations(self.component,size)
        for mix in mixes :
            for combo in product(*mix):
                if self.check(combo) :
                    moves.append(combo)
        return moves
    def check(self,combo):
        for n in combo :
            if self.graph.nodes[n]['type'] != 'terminal' :
                return False
            
        for n in combo :
            if self.graph.degree[n] >= self.dim:
                return False
        return True


    def get_bifurcation(self):
        moves = []
        for u,v in self.graph.edges:
            comp = self.track_connected[u]
            for other_comp in self.component :
                if comp!=other_comp :
                    for k in other_comp :
                        moves.append((u,v,k))
        return moves

    def play_move(self, move):
        """
        Play a legal move by adding edges or a Steiner point.
        """
        if len(move) == 2:
            # Add an edge between two points
            p1, p2 = move
            self.graph.add_edge(p1, p2, weight=distance.euclidean(p1, p2))
        else:
            # Add a Steiner point for triplet (2D)
            centroid = tuple(np.mean(move, axis=0))
            self.graph.add_node(centroid, type='steiner')
            for point in move:
                self.graph.add_edge(centroid, point, weight=distance.euclidean(centroid, point))
            self.graph.remove_edge(move[0],move[1])
        # Update connected components
        new_component = set()

        if len(move)>2 : #Steiner Point addition case
            new_component.update(self.track_connected[move[0]])
            new_component.update(self.track_connected[move[-1]])
            self.component.remove(self.track_connected[move[0]])
            self.component.remove(self.track_connected[move[-1]])
            new_component.add(centroid)
        else : #Other case
            for point in move:
                new_component.update(self.track_connected[point])
                self.component.remove(self.track_connected[point])
        self.component.append(new_component)
        for point in new_component:
            self.track_connected[point] = new_component
    
    def terminal(self):
        return len(self.component)<=1
    
    def get_hash(self):
        return nx.weisfeiler_lehman_graph_hash(self.graph)

    def optimize(self):
        pass
    
    def get_score(self):
        if self.terminal :
            self.optimize()
            return sum(nx.get_edge_attributes(self.graph, 'weight').values())
        else :
            return None
        
    def plot_tree(self):
        """
        Plot the Steiner tree with correct node positions.
        """
        pos = { }
        for node in self.graph.nodes :
            pos[node] = node

        node_colors = ['red' if self.graph.nodes[n]['type'] == 'terminal' else 'blue' for n in self.graph.nodes]
        
        plt.figure(figsize=(8, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color='black')
        plt.show()
    
if __name__ == "__main__":
    # Define a simple test case: a square with a central Steiner point
    terminals = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    tree = EuclideanSteinerTree(terminals)
    terminals = np.array([(0, 0), (1, 0),(0, 1), (1, 1)])
    tree2 = EuclideanSteinerTree(terminals)

    print(nx.weisfeiler_lehman_graph_hash(tree.graph))
    print(nx.weisfeiler_lehman_graph_hash(tree2.graph))

    
    print("Initial legal moves:")
    for move in tree.legal_moves():
        print(move)
    tree.play_move(tree.legal_moves()[-1])
    tree.plot_tree()

    print("########################")

    for move in tree.legal_moves():
        print(move)
    tree.play_move(tree.legal_moves()[-1])
    tree.plot_tree()

    print("########################")

    for move in tree.legal_moves():
        print(move)
    tree.play_move(tree.legal_moves()[-3])
    tree.plot_tree()

    print("########################")
    print(tree.component)
    print(tree.legal_moves())
    

## Ajoute correction de pt de steiner
## Ajoute score
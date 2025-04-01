import numpy as np
import networkx as nx
from scipy.spatial import distance
from itertools import product,combinations
import matplotlib.pyplot as plt

class EuclideanSteinerTree:

    def __init__(self, points: np.ndarray):
        """
        Initialize the Steiner Tree with terminal points and compute minimum spanning tree.
        :param points: A numpy array of shape (N, 2) or (N, 3) representing terminal points in 2D or 3D.
        """
        self.dim = points.shape[1]  # Determine if 2D or 3D
        self.max_degree = self.dim + 1
        self.graph = nx.Graph()
        
        # Add terminal nodes to the graph
        for point_idx in range(points.shape[0]):
            self.graph.add_node(point_idx, type='terminal',position=points[point_idx])
        
        # Connect all nodes with edges weighted by Euclidean distance
        for i in range(points.shape[0]-1):
            for j in range(i,points.shape[0]):
                    # Add edge with distance as weight
                    self.graph.add_edge(i, j, weight=np.linalg.norm((points[i]-points[j])))
        
        # Compute minimum spanning tree
        self.graph = nx.minimum_spanning_tree(self.graph, weight='weight')

    def get_merges(self):
        merge_moves = []
        for node in self.graph :
            # Get all terminal nodes with a degree >=2
            if (self.graph.degree[node]>= self.dim) and (self.graph.nodes[node]['type']=='terminal'):
                merges = []
                neighbors = [t[-1] for t in self.graph.edges(node)]
                mixes = combinations(neighbors,2)
                for mix in mixes :
                    merges.append(mix)
        
                for merge in merges:
                    #merge_moves.append(sorted(list(merge)+[node], key=lambda x: (x,) if isinstance(x, int) else x))
                    # Place the node with high degree at the end !
                    #TODO : We can check here if Fermat
                    merge_moves.append(list(merge)+[node])
        return merge_moves

    def get_swaps(self):
        return self.graph.egdes()

    def legal_moves(self):
        moves = []
        moves += self.get_swaps()
        moves += self.get_merges()
        return moves
    
    def play_move(self, move):
        if len(move)==2 :
            self.play_swap(move)
        
        elif len(move)==3 :
            self.replace_triangle_with_steiner_point(move)
        else :
            raise ValueError


    def replace_triangle_with_steiner_point(self, nodes_list):

        assert len(nodes_list) == 3, "Must provide exactly 3 nodes"
        
        # Get the last node and the other two nodes
        node3 = nodes_list[-1]  # Remove and get the last element
        node1, node2 = nodes_list[0],nodes_list[1]  # The remaining two nodes
        
        self.graph.remove_edge(node1, node3)
        self.graph.remove_edge(node2, node3)
        
        # Calculate the optimal Steiner point position (for 3 points, it's the Fermat point)
        pos1 = self.graph.nodes[node1]['position']
        pos2 = self.graph.nodes[node2]['position']
        pos3 = self.graph.nodes[node3]['position']
        
        # For init, let's use the geometric center (centroid)
        steiner_pos = tuple(np.mean([pos1, pos2, pos3], axis=0))
        
        # Create a new node ID for the Steiner point
        new_node_id = tuple(sorted(nodes_list, key=lambda x: (x,) if isinstance(x, int) else x))
        
        # Add the new Steiner point to the graph
        self.graph.add_node(new_node_id, type='steiner', position=steiner_pos)
        
        # Connect the Steiner point to all three original nodes
        for node in [node1, node2, node3]:
            pos = self.graph.nodes[node]['position'] 
            distance = np.linalg.norm((steiner_pos- pos))
            self.graph.add_edge(new_node_id, node, weight=distance)

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
        plt.figure(figsize=(8, 8))
        for n in self.graph.nodes :
            x,y = self.graph.nodes[n]['position'][0], self.graph.nodes[n]['position'][1]
            c = 'red' if self.graph.nodes[n]['type'] == 'terminal' else 'blue'
            plt.scatter(x,y,c=c)
            plt.annotate(str(n),(x,y))

        for e in self.graph.edges():
            pos_0 = self.graph.nodes[e[0]]['position'] 
            pos_1 = self.graph.nodes[e[1]]['position'] 
            plt.plot([pos_0[0],pos_1[0]],[pos_0[1],pos_1[1]],c='black')
        plt.show()
    
if __name__ == "__main__":
    # Define a simple test case: a square with a central Steiner point
    terminals = np.array([(0, 0), (0, 1), (1, 0), (1, 1)],dtype=np.float32)
    tree = EuclideanSteinerTree(terminals)
    tree.plot_tree()
    tree.play_move(tree.get_merges()[0])
    tree.plot_tree()
    print(nx.weisfeiler_lehman_graph_hash(tree.graph))
    

## Ajoute correction de pt de steiner
## Ajoute score
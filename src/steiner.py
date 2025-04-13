import numpy as np
import networkx as nx
from scipy.spatial import distance
from itertools import product,combinations
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

#from torchmin import minimize
#import torch

class EuclideanSteinerTree:

    def __init__(self, points: np.ndarray,naming=None):
        """
        Initialize the Steiner Tree with terminal points and compute minimum spanning tree.
        :param points: A numpy array of shape (N, 2) or (N, 3) representing terminal points in 2D or 3D.
        """
        assert points.shape[1] == 2 #(Now only works on 2D)
        self.dim = points.shape[1]  # Determine if 2D or 3D 
        self.max_degree = self.dim + 1
        self.graph = nx.Graph()
        self.naming = naming
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
        self.MST_score = self.get_score()
        self.terminal = False


    def get_merges(self):
        merge_moves = []
        for node in self.graph :
            # Get all terminal nodes with a degree >=2
            if (self.graph.degree[node]>= self.dim) and (self.graph.nodes[node]['type']=='terminal'):
                merges = []
                neighbors = sorted([t[-1] for t in self.graph.edges(node)])
                mixes = combinations(neighbors,2)
                for mix in mixes :
                    merges.append(mix)
        
                for merge in merges:
                    #merge_moves.append(sorted(list(merge)+[node], key=lambda x: (x,) if isinstance(x, int) else x))
                    # Place the node with high degree at the end !
                    #TODO : We can check here if Fermat I mean not mandatory could even be a bad idea
                    merge_moves.append(list(merge)+[node])
        return merge_moves

    def get_swaps(self):
        swap_moves = []
        for e in self.graph.edges():
            if ((self.graph.nodes[e[0]]['type']=='terminal') 
                or (self.graph.nodes[e[1]]['type']=='terminal')) :
                    
                    graph = self.graph.copy()
                    graph.remove_edge(*e)
                    components = list(nx.connected_components(graph))
                    assert len(components) == 2

                    replace_by = product(components[0],components[1])
                    for r in replace_by :
                        r = tuple(sorted(r))
                        if (e[0]!=r[0]) or (e[1]!=r[1]):
                            if ((graph.degree[r[0]]<2) and (graph.degree[r[1]]<2)):
                                swap_moves.append((e,r))

        return swap_moves

    def legal_moves(self):
        if self.terminal : return []
        moves = []
        moves += self.get_swaps()
        moves += self.get_merges()
        moves += ['STOP']
        return moves
    
    def play_move(self, move):
        if len(move)==2 :
            self.play_swap(move)
        elif len(move)==3 :
            self.replace_triangle_with_steiner_point(move)
        elif move == 'STOP':
            self.terminal = True
            pass
        else :
            raise ValueError

    def separate_steiner_connection(self,steiner,terminal):

        neighbors = [t[-1] for t in self.graph.edges(steiner)]
        assert (len(neighbors)==3) and (terminal in neighbors)
        t_idx = neighbors.index(terminal)
        neighbors.pop(t_idx)

        node1, node2 = neighbors
        pos_1 = self.graph.nodes[node1]['position']  
        pos_2  = self.graph.nodes[node2]['position']
        distance = np.linalg.norm((pos_1- pos_2))
        self.graph.add_edge(node1, node2, weight=distance)

        self.graph.remove_node(steiner)

    def add_new_edge(self,to_replace):
        pos_1 = self.graph.nodes[to_replace[0]]['position']  
        pos_2  = self.graph.nodes[to_replace[1]]['position']
        distance = np.linalg.norm((pos_1- pos_2))
        self.graph.add_edge(to_replace[0], to_replace[1], weight=distance)

    def play_swap(self,move):
        assert len(move)==2
        to_remove,to_replace = move

        if  ((self.graph.nodes[to_remove[0]]['type']=='terminal') 
             and (self.graph.nodes[to_remove[1]]['type']=='terminal')):
            self.graph.remove_edge(to_remove[0], to_remove[1])
            self.add_new_edge(to_replace)

        elif (self.graph.nodes[to_remove[0]]['type']=='steiner'):
            assert (self.graph.nodes[to_remove[1]]['type']=='terminal')
            steiner,terminal = to_remove[0],to_remove[1]
            self.separate_steiner_connection(steiner,terminal)
            self.add_new_edge(to_replace)

        elif (self.graph.nodes[to_remove[1]]['type']=='steiner'):
            assert (self.graph.nodes[to_remove[0]]['type']=='terminal')
            steiner,terminal = to_remove[1],to_remove[0]
            self.separate_steiner_connection(steiner,terminal)
            self.add_new_edge(to_replace)
            
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
        steiner_pos = np.mean([pos1, pos2, pos3], axis=0)
        
        # Create a new node ID for the Steiner point (Choose anaming conv)
        if self.naming :
            new_node_id = tuple(sorted(nodes_list, key=lambda x: (x,) if isinstance(x, int) else x))
        else : new_node_id = max(self.graph.nodes)+1
        
        # Add the new Steiner point to the graph
        self.graph.add_node(new_node_id, type='steiner', position=steiner_pos)
        
        # Connect the Steiner point to all three original nodes
        for node in [node1, node2, node3]:
            pos = self.graph.nodes[node]['position'] 
            distance = np.linalg.norm((steiner_pos- pos))
            self.graph.add_edge(new_node_id, node, weight=distance)
        
        self.optimize()

    def get_weisfeiler_lehman_graph_hash(self):
        return nx.weisfeiler_lehman_graph_hash(self.graph)
    
    def get_hash(self):
        hash_list= []
        for e in self.graph.edges():
            tuple_list = []
            t_0 = self.graph.nodes[e[0]]['position']
            t_1 = self.graph.nodes[e[1]]['position']
            for pos in [t_0,t_1]:
                tuple_list.append(tuple(round(float(x), 4) for x in pos))
            edge = tuple(sorted(tuple_list))
            hash_list.append(edge)
        tuple_to_hash = tuple(sorted(hash_list))
        return hash(tuple_to_hash)

    def optimize(self):
        variables_dict = {}
        idx = 0

        ### Find Variables AKA Steiner points
        for node in self.graph:
            if (self.graph.nodes[node]['type']=='steiner'):
                variables_dict[node]=idx
                idx+=1

        ### Init their position Array
        var_array = np.zeros((idx,2),np.float32)
        for node,i in variables_dict.items():
            var_array[i] = self.graph.nodes[node]['position']

        relevant_connexion_st = []
        relevant_connexion_ss = []

        for e in self.graph.edges():
            t_0 = self.graph.nodes[e[0]]['type']
            t_1 = self.graph.nodes[e[1]]['type']

            if ((t_0=='steiner') or (t_1=='steiner')):
                if (t_0=='steiner'):
                    if (t_1=='steiner'):
                        connexion = variables_dict[e[0]],variables_dict[e[1]]
                        #Steiner - Steiner
                        relevant_connexion_ss.append(connexion) 
                    else :
                        connexion = variables_dict[e[0]],self.graph.nodes[e[1]]['position']
                        #Steiner - Terminal
                        relevant_connexion_st.append(connexion)
                else :
                    if (t_0=='steiner'):
                        connexion = variables_dict[e[0]],variables_dict[e[1]]
                        #Steiner - Steiner
                        relevant_connexion_ss.append(connexion) 
                    else :
                        connexion = variables_dict[e[1]],self.graph.nodes[e[0]]['position']
                        #Steiner - Terminal
                        relevant_connexion_st.append(connexion)

        function_to_minimize = create_problem(relevant_connexion_st,
                                                   relevant_connexion_ss,
                                                   var_array.shape[1])
        
        result = minimize(
            function_to_minimize, 
            var_array.flatten(), 
            method='BFGS',
            options={'eps': 1e-10,'maxiter':1000}
        )
        
        # Reshape the optimized positions
        optimized_positions = result.x.reshape((-1,2))
        for node,i in variables_dict.items():
            self.graph.nodes[node]['position'] = optimized_positions[i]
        
        # Update weights
        for e in self.graph.edges():
            t_0 = self.graph.nodes[e[0]]['type']
            t_1 = self.graph.nodes[e[1]]['type']

            if ((t_0=='steiner') or (t_1=='steiner')):
                pos0 = self.graph.nodes[e[0]]['position']
                pos1 = self.graph.nodes[e[1]]['position']
                dist = np.linalg.norm((pos0-pos1)) 
                self.graph[e[0]][e[1]]['weight'] = dist
        

    
    def get_score(self):
        return sum(nx.get_edge_attributes(self.graph, 'weight').values())

    def get_normalized_score(self):
        return (1-(self.get_score()/self.MST_score)) #/ 7.45  see paper
        
    def fill_plot(self,ax):
        ax.clear()
        for n in self.graph.nodes :
            x,y = self.graph.nodes[n]['position'][0], self.graph.nodes[n]['position'][1]
            c = 'red' if self.graph.nodes[n]['type'] == 'terminal' else 'blue'
            ax.scatter(x,y,c=c,s=100)
            ax.annotate(str(n),(x,y),fontsize=6, fontweight='bold'
                        ,bbox=dict(facecolor='white', alpha=0.4),
                        textcoords="offset points", xytext=(8,5), ha='center')

        for e in self.graph.edges():
            pos_0 = self.graph.nodes[e[0]]['position'] 
            pos_1 = self.graph.nodes[e[1]]['position'] 
            ax.plot([pos_0[0],pos_1[0]],[pos_0[1],pos_1[1]],c='black')
        title = "Score : "+str(self.get_score())
        ax.set_title(title)
        
    def plot_tree(self,edge_only = False):
        """
        Plot the Steiner tree with correct node positions.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        if edge_only : self.fill_edge_only(ax)
        else : self.fill_plot(ax)

        plt.show()

    def manually_load_steiner_points(self,steiner_points):
        for point_idx in range(steiner_points.shape[0]):
            self.graph.add_node(point_idx+max(self.graph.nodes)+1, type='steiner',position=steiner_points[point_idx])
        for i in self.graph:
            for j in self.graph:
                    # Add edge with distance as weight
                    if i!=j :
                        pi = self.graph.nodes[i]['position'] 
                        pj = self.graph.nodes[j]['position'] 
                        self.graph.add_edge(i, j, weight=np.linalg.norm((pi-pj)))
        self.graph = nx.minimum_spanning_tree(self.graph, weight='weight')
    
    def fill_edge_only(self,ax,visible=True):
        lines = []
        for e in self.graph.edges() :
            pos_0 = self.graph.nodes[e[0]]["position"]
            pos_1 = self.graph.nodes[e[1]]["position"]
            line, = ax.plot([pos_0[0], pos_1[0]], [pos_0[1], pos_1[1]], 
                            c="green", linewidth=2, visible=visible,linestyle="dotted")
            lines.append(line)
        return lines
    
    def playout(self,max_iter = 10):
        end = False
        iter = 0
        move_list = []
        move_idx_list = []
        while (not end) and (iter < max_iter):
            moves = self.legal_moves()
            n = random.randint (0, len (moves) - 1)
            if n == (len(moves)-1):
                end = True
            else : 
                self.play_move(moves[n])
            move_list.append(moves[n])
            move_idx_list.append(n)
            iter+=1
        return move_list,move_idx_list


def create_problem(relevant_connexion_st,relevant_connexion_ss,jump):
    def function_to_minimize(X):
        sum = 0
        for connexion in relevant_connexion_st:
            sum+= np.linalg.norm(
                ((X[connexion[0]*jump:connexion[0]*jump+2])-
                 connexion[1])
                )
        for connexion in relevant_connexion_ss :
            sum+= np.linalg.norm(
                (X[connexion[0]*jump:connexion[0]*jump+2] - 
                 X[connexion[1]*jump:connexion[1]*jump+2])
            )
        return sum
    return function_to_minimize

    
if __name__ == "__main__":
    # Define a simple test case: a square with a central Steiner point
    terminals = np.array([(0., 0),(0, 1), (1, 0), (1, 1)])#,(2, 0.5)],dtype=np.float32)
    tree = EuclideanSteinerTree(terminals)
    tree.plot_tree()

    tree.play_move(tree.legal_moves()[-1])
    tree.plot_tree()
    tree.play_move(tree.legal_moves()[-1])
    print(tree.get_hashV2())
    tree.plot_tree(edge_only=True)
    tree.playout()
    #print(nx.weisfeiler_lehman_graph_hash(tree.graph))
    #print(tree.get_swaps())
    pass

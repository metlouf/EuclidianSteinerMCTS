from src.steiner import EuclideanSteinerTree
import numpy as np
from src.orlib_loader import load_problem_file,load_solution_file
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import random


class TranspositionTable():
    """Dictionnary where each entry correspond to the hash of a board."""
    def __init__(self):
        self.Table = {}
    
    def add(self,tree : EuclideanSteinerTree):
        scores = [-10e10 for _ in range(len(tree.legal_moves()))]
        playouts = [0.0 for _ in range(len(tree.legal_moves()))]
        self.Table[tree.get_hash()] = [0, playouts, scores]

    def look(self,tree):
        return self.Table.get(tree.get_hash(), None)

def UCT_best_move(tree : EuclideanSteinerTree, Table : TranspositionTable,depth,max_depth,C = 0.004):

    if depth == max_depth :
        return tree.get_normalized_score()
    
    t = Table.look(tree)
    if t != None:
        legal_moves = tree.legal_moves()

        best_move = legal_moves[0]
        best_idx = 0
        best_value = 0
        
        enum = enumerate(legal_moves)
        
        for (idx,move) in enum:
            n = t[0]
            ni = t[1][idx]
            si = t[2][idx]
            value = 10e10
            
            if ni > 0:
                value = si + C * np.sqrt(np.log (n) / ni)
                 
            if value > best_value:
                best_value = value
                best_move = move
                best_idx = idx
        if len(best_move) >2 :
            pass
        tree.play_move(best_move)
        depth_tmp = depth+1
        
        res = UCT_best_move(tree,Table,depth_tmp,max_depth)
        t[0] += 1
        t[1][best_idx] += 1
        t[2][best_idx] = np.max([res,t[2][best_idx]])

        return t[2][best_idx]

    else :
        Table.add(tree)
        tree.playout(max_iter=0)
        return tree.get_normalized_score()

def UCT(tree : EuclideanSteinerTree, max_depth = 1e9, num_sim = 10, Verbose = True,C = 0.004):
    """ UCT code inspired by the one given in class"""
    
    depth = 0
    Table = TranspositionTable()
        
        
    for _ in tqdm(range(num_sim)) :
        test_tree = copy.deepcopy(tree)
        score = UCT_best_move(test_tree,Table,depth,max_depth,C)
        
    t = Table.look(tree)
    
    legal_moves = tree.legal_moves()
    moves = []
    indexes  = []

    best_move = "STOP"
    best_score = -10e10
    best_idx = 1

    while True and (depth < max_depth) :
        enum = enumerate(legal_moves)
        for (idx,move) in enum:
            if (t[2][idx] >= best_score):
                best_score = t[2][idx]
                best_move = move

        if best_move!="STOP":

            tree.play_move(best_move)
            moves.append(best_move)
            indexes.append(best_idx)

            ## Back to original state
            best_move = "STOP"
            best_idx = "q"
            t = Table.look(tree)
            legal_moves = tree.legal_moves()
        else :
            return best_score,moves,indexes
    return best_score,moves,indexes


if __name__ == "__main__":

    chosen = 20
    chosen_index = 3

    problem_file = f"data/estein{chosen}.txt"
    solution_file = f"data/estein{chosen}opt.txt"

    list_of_problems = load_problem_file(problem_file)
    list_of_solutions = load_solution_file(solution_file)

    terminals = np.array(list_of_problems[chosen_index],dtype=np.float32)

    solution_tree = EuclideanSteinerTree(terminals)
    solution_tree.manually_load_steiner_points(
        np.array(list_of_solutions[chosen_index]['steiner_points'])
    )
    print("Best Theoritical Score :",solution_tree.get_normalized_score())

    tree = EuclideanSteinerTree(terminals)
    
    score,moves,indexes = UCT(tree,Verbose=False,max_depth=6,num_sim=1000)
    print("UCT :",tree.get_normalized_score())

    fig, ax = plt.subplots(figsize=(8, 8))
    
    tree.fill_plot(ax)
    solution_lines = solution_tree.fill_edge_only(ax)
    ax.set_title(f"{tree.get_normalized_score()} -- Best score {solution_tree.get_normalized_score()}")
    plt.show()

    merges = 0
    swaps = 0
    for k in moves : 
        if len(k)==3 : merges+=1
        else : swaps+=1
    print(f"{merges} Merges and {swaps} Swaps out of {len(moves)} Moves")

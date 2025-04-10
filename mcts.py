from src.steiner import EuclideanSteinerTree
import numpy as np
from src.orlib_loader import load_problem_file,load_solution_file
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import random

def eval_move(args):
    i,tree,move = args
    test_tree = copy.deepcopy(tree)
    test_tree.play_move(move)
    score = test_tree.get_normalized_score()
    return score

def greedy_search(tree : EuclideanSteinerTree,Verbose = False,max_depth = 1e9,parallel = False):

    best_score = tree.get_normalized_score()
    legal_moves = tree.legal_moves()

    best_move = "STOP"
    best_idx = "q"

    moves = []
    indexes  = []
    depth = 0

    while True and (depth < max_depth) :
        depth+=1

        move_args = [(i,tree, move) for i,move in enumerate(legal_moves)]
        random.shuffle(move_args)
        if parallel :
            chunksize = max(1, len(move_args) // (os.cpu_count() * 4))
            results = process_map(eval_move, move_args,chunksize = chunksize)

        else :
            results = [0]*len(move_args)
            enum = enumerate(move_args)
            if Verbose : enum = enumerate(tqdm(move_args))
            for idx,move_args in enum :
                score = eval_move(move_args)
                results[idx] = score

        for idx,score in enumerate(results):
            if score > best_score :
                best_score = score
                i,_, move = move_args[idx]
                best_move = move
                best_idx = i+1
        
        if best_move!="STOP":
            tree.play_move(best_move)
            moves.append(best_move)
            indexes.append(best_idx)

            ## Back to original state
            best_move = "STOP"
            best_idx = "q"
            legal_moves = tree.legal_moves()

        else :
            return best_score,moves,indexes
    return best_score,moves,indexes

def monte_carlo_scoring(tree: EuclideanSteinerTree,num_sim :int):
    test_tree = copy.deepcopy(tree)
    mean_score = 0
    for i in tqdm(range(num_sim),desc="Playout"):
        test_tree.playout()
        score = test_tree.get_normalized_score()
        mean_score+=score
    return mean_score/num_sim

def monte_carlo_eval_move(args):pass
def flat_monte_carlo_search(tree : EuclideanSteinerTree,
                            Verbose = False,max_depth = 1e9,num_sim = 100,parallel = False):

    legal_moves = tree.legal_moves()

    ## Score Stopping 

    best_score = monte_carlo_scoring(tree,num_sim)
    best_move = "STOP"
    best_idx = "q"
    

    moves = []
    indexes  = []
    depth = 0

    while True and (depth < max_depth) :
        depth+=1

        move_args = [(tree, move) for move in legal_moves]
        if parallel :
            chunksize = max(1, len(move_args) // (os.cpu_count() * 2))
            results = process_map(eval_move, move_args,chunksize = chunksize)


        enum = enumerate(legal_moves)
        if Verbose : enum = enumerate(tqdm(legal_moves))
        for idx,move in enum :
            test_tree = copy.deepcopy(tree)
            test_tree.play_move(move)
            mean_score = monte_carlo_scoring(test_tree,num_sim)
            if mean_score > best_score :
                best_move = move
                best_score = mean_score
                best_idx = idx+1
        
        if best_move!="STOP":

            tree.play_move(best_move)
            moves.append(best_move)
            indexes.append(best_idx)

            ## Back to original state
            best_move = "STOP"
            best_idx = "q"
            legal_moves = tree.legal_moves()
        else :
            return best_score,moves,indexes
    return best_score,moves,indexes


class TranspositionTable():
    """Dictionnary where each entry correspond to the hash of a board."""
    def __init__(self):
        self.Table = {}
    
    def add(self,tree : EuclideanSteinerTree):
        scores = [0.0 for _ in range(len(tree.legal_moves()))]
        playouts = [0.0 for _ in range(len(tree.legal_moves()))]
        self.Table[tree.get_hash()] = [0, playouts, scores]

    def look(self,tree):
        return self.Table.get(tree.get_hash(), None)


def UCT_best_move(tree : EuclideanSteinerTree, Table : TranspositionTable,depth,max_depth):
    if depth == max_depth :
        return tree.get_normalized_score()
    t = Table.look(tree)
    if t != None:
        legal_moves = tree.legal_moves()
        best_move = legal_moves[0]
        best_idx = 1
        best_value = 0
        
        enum = enumerate(legal_moves)
        
        for (idx,move) in enum:
            n = t[0]
            ni = t[1][idx]
            si = t[2][idx]
            value = -10
            
            if ni > 0:
                value = si/ni + 0.4 * np.sqrt(np.log (n) / ni)
                 
            if value > best_value:
                best_value = value
                best_move = move
                best_idx = idx+1
         
        tree.play_move(best_move)
        depth_tmp = depth+1
        
        res = UCT_best_move(tree,Table,depth_tmp,max_depth)
        t[0] += 1
        t[1][best_idx] += 1
        t[2][best_idx] += res
        return res

    else :
        Table.add(tree)
        tree.playout()
        return tree.get_normalized_score()

def UCT(tree : EuclideanSteinerTree, max_depth = 1e9, num_sim = 10, Verbose = True):
    """ UCT code inspired by the one given in class"""
    
    best_score = tree.get_normalized_score()
    legal_moves = tree.legal_moves()

    moves = []
    indexes  = []
    depth = 0
    Table = TranspositionTable()
    while True and (depth < max_depth) :  
        depth+= 1
        enum = enumerate(legal_moves)
        if Verbose : enum = enumerate(tqdm(legal_moves))
        best_value = -100
        
        
        for _ in tqdm(range(num_sim)) :
            test_tree = copy.deepcopy(tree)
            score = UCT_best_move(test_tree,Table,depth,max_depth)
            
        
        t = Table.look(tree)
        
        legal_moves = tree.legal_moves()

        best_move = "STOP"
        best_value = 0
        best_idx = 1
        enum = enumerate(legal_moves)
        if Verbose : enum = enumerate(tqdm(legal_moves))
        for (idx,move) in enum:
            if (t[1][idx] > best_value):
                best_value = t[1][idx]
                best_move = move
        
        if best_move!="STOP":

            tree.play_move(best_move)
            moves.append(best_move)
            indexes.append(best_idx)

            ## Back to original state
            best_move = "STOP"
            best_idx = "q"
            legal_moves = tree.legal_moves()
        else :
            return best_score,moves,indexes
    return best_score,moves,indexes


if __name__ == "__main__":

    chosen = 10
    chosen_index = 2

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

    #score,moves,indexes = greedy_search(tree,Verbose=True,max_depth=10,parallel=True)
    #print("Greedy Score :",score)

    #score,moves,indexes = flat_monte_carlo_search(tree,Verbose=True,max_depth=10,num_sim=10)
    #print("Monte Carlo Score :",score)
    
    score,moves,indexes = UCT(tree,Verbose=False,max_depth=6,num_sim=100)
    print("UCT :",score)

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

        








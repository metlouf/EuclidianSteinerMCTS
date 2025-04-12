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
"""
def monte_carlo_scoring(tree: EuclideanSteinerTree,num_sim :int,Verbose = False):
    mean_score = 0
    if Verbose : rg = tqdm(range(num_sim),desc="Playout")
    else : rg = range(num_sim)
    for i in rg :
        test_tree = copy.deepcopy(tree)
        test_tree.playout()
        score = test_tree.get_normalized_score()
        mean_score+=score
    return mean_score/num_sim

def monte_carlo_eval_move(args):
    i,tree,move,num_sim,verbose = args
    test_tree = copy.deepcopy(tree)
    test_tree.play_move(move)
    mean_score = monte_carlo_scoring(test_tree,num_sim,Verbose=verbose)
    return mean_score
"""

def monte_carlo_eval_move(tree):
    test_tree = copy.deepcopy(tree)
    move_list,move_idx_list = test_tree.playout()
    return test_tree.get_normalized_score(),move_list,move_idx_list

def flat_monte_carlo_search(tree : EuclideanSteinerTree,
                            Verbose = False,num_sim = 1000000,parallel = False):
    
    best_score = 0
    best_move = []
    best_idx = []

    for i in tqdm(range(num_sim)):
        s,ml,mil = monte_carlo_eval_move(tree)
        if s>best_score:
            best_score=s
            best_move,best_idx = ml,mil
            print("best so far",s)
        
    return best_score,best_move,best_idx


if __name__ == "__main__":

    chosen = 20
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

    score,moves,indexes = flat_monte_carlo_search(tree,Verbose=False,parallel=False)
    print("Monte Carlo Score :",score)
    

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

        








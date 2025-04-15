from src.steiner import EuclideanSteinerTree
import numpy as np
from src.orlib_loader import load_problem_file,load_solution_file
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import random
import json
import argparse
import datetime

## Code Inspired by https://www.lamsade.dauphine.fr/~cazenave/Breakthrough.ipynb

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
        try :
            legal_moves = tree.legal_moves()

            best_move = "STOP"
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
        except :
            return tree.get_normalized_score()
    else :
        Table.add(tree)
        tree.playout(max_iter=0)
        return tree.get_normalized_score()

def UCT(tree : EuclideanSteinerTree, max_depth = 1e9, num_sim = 10, Verbose = True,C = 0.01):
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
    best_idx = -1

    while True and (depth < max_depth) :
        enum = enumerate(legal_moves)
        depth+=1
        for (idx,move) in enum:
            if (t[2][idx] > best_score):
                best_score = t[2][idx]
                best_move = move
                best_idx = idx+1

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

def main():
    parser = argparse.ArgumentParser(description="Steiner Tree Optimization")
    
    parser.add_argument('--chosen', type=int, help='Chosen problem number (e.g., 10)',default=10)
    parser.add_argument('--chosen_index', type=int, help='Index inside the problem list',default=2)
    parser.add_argument('--C', type=float,default=0.4)
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth for greedy search')
    parser.add_argument('--num_sim', type=int, default=1000, help='Max depth for greedy search')
    parser.add_argument('--parallel', action='store_true', help='Enable parallelism for greedy search')

    args = parser.parse_args()

    # Load problem and solution
    problem_file = f"data/estein{args.chosen}.txt"
    solution_file = f"data/estein{args.chosen}opt.txt"

    list_of_problems = load_problem_file(problem_file)
    list_of_solutions = load_solution_file(solution_file)

    terminals = np.array(list_of_problems[args.chosen_index], dtype=np.float32)

    solution_tree = EuclideanSteinerTree(terminals)
    solution_tree.manually_load_steiner_points(
        np.array(list_of_solutions[args.chosen_index]['steiner_points'])
    )

    best_theoretical_score = solution_tree.get_normalized_score()
    print("Best Theoretical Score:", best_theoretical_score)

    tree = EuclideanSteinerTree(terminals)


    score, moves, indexes = UCT(tree,Verbose=False,max_depth=7,num_sim=args.num_sim,C=args.C)

    print(f"UCT Score:", score)
    '''
    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 8))
    tree.fill_plot(ax)
    solution_tree.fill_edge_only(ax)
    ax.set_title(f"{tree.get_normalized_score()} -- Best score {solution_tree.get_normalized_score()}")
    plt.show()
    '''
    # Analyze moves
    merges = sum(1 for k in moves if len(k) == 3)
    swaps = len(moves) - merges

    print(f"{merges} Merges and {swaps} Swaps out of {len(moves)} Moves")

    # Save results
    result_data = {
        "mode": "UCT",
        "chosen": args.chosen,
        "chosen_index": args.chosen_index,
        "score": score,
        "best_theoretical_score": best_theoretical_score,
        "merges": merges,
        "swaps": swaps,
        "total_moves": len(moves),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    output = f"result100K/result_{args.chosen}_{args.chosen_index}_UCT.json"
    if output.endswith(".json"):
        with open(output, 'w') as f:
            json.dump(result_data, f, indent=4)

if __name__ == "__main__":
    main()
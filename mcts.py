from src.steiner import EuclideanSteinerTree
import numpy as np
from src.orlib_loader import load_problem_file,load_solution_file
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

def greedy_search(tree : EuclideanSteinerTree,Verbose = False,max_depth = 1e9):

    best_score = tree.get_normalized_score()
    legal_moves = tree.legal_moves()

    best_move = "STOP"
    best_idx = "q"

    moves = []
    indexes  = []
    depth = 0

    while True and (depth < max_depth) :
        depth+=1
        enum = enumerate(legal_moves)
        if Verbose : enum = tqdm(enum)
        for idx,move in enum :
            test_tree = copy.deepcopy(tree)
            test_tree.play_move(move)
            score = test_tree.get_normalized_score()
            if score > best_score :
                best_move = move
                best_score = score
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


if __name__ == "__main__":

    chosen = 60
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
    score,moves,indexes = greedy_search(tree,Verbose=True,max_depth=10)

    print("Greedy Score :",score)

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

        








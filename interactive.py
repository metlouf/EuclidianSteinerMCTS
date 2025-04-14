import matplotlib.pyplot as plt


def play_interactive_game(tree,solution_tree):
    """
    Play the tree game interactively in the terminal.
    
    Args:
        tree: Your tree game object with methods like legal_moves(), play_move(), and plot_tree()
    """
    print("Welcome to the Interactive Steiner Tree Game!")
    print("-----------------------------------")
    
    game_over = False
    solution_visible = [True]
    
    def on_click(event):
        solution_visible[0] = not solution_visible[0]   # Toggle state
        for line in solution_lines:
            line.set_visible(solution_visible[0])  # Show/hide solution
        plt.draw()


    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.mpl_connect("button_press_event", on_click)



    ax.set_aspect('equal')

    # Display current state
    print("\nCurrent Tree :")
    tree.fill_plot(ax)
    solution_lines = solution_tree.fill_edge_only(ax,visible=solution_visible[0])
    ax.set_title(ax.get_title()+f" -- Best score {solution_tree.get_score()}")
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel(f'Hash :{tree.get_hash()}')
    plt.draw()
    
    init_score = tree.get_score()
    print("Score :", init_score)

    while not game_over:

        
        # Get legal moves
        moves = tree.legal_moves()
  
        # Display available moves with index numbers
        print("\nAvailable moves:")
        for i, move in enumerate(moves, 1):
            mode = "Swap" if len(move)==2 else "Merge"
            print(f"{i}. {mode} {move}")

        if not moves:
            print("\nGame ended.")
            break
        
        # Get user input (just the index number)
        while True:
            try:
                choice = input("\nEnter move number (or 'q' to quit): ")
                
                if choice.lower() == 'q':
                    game_over = True
                    print("Thanks for playing!")
                    break
                
                move_index = int(choice) - 1
                if 0 <= move_index < len(moves):
                    selected_move = moves[move_index]
                    break
                else:
                    print(f"Invalid choice. Please select a number between 1 and {len(moves)}.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
        
        if game_over:
            break
        
        # Play the selected move
        print(f"\nPlaying move: {selected_move}")
        tree.play_move(selected_move)
        tree.fill_plot(ax)
        solution_lines = solution_tree.fill_edge_only(ax,visible=solution_visible[0])
        ax.set_title(ax.get_title()+f" -- Best score {solution_tree.get_score()}")
        ax.xaxis.set_label_position('bottom')
        ax.set_xlabel(f'Hash :{tree.get_hash()}')#+f'\nHashV2 :{tree.get_hashV2()}')
        plt.draw()
        print("Score :", tree.get_score())
        
    # Show final state
    print("\nFinal Score:")
    final_score = tree.get_score()
    print(final_score)

    if init_score > final_score:
        rho =(final_score)/init_score
        print("You win ! Ratio =",rho)
    else :
        print("NOOB")

    print("\nGame ended.")

# Example usage:
if __name__ == "__main__":

    from src.steiner import EuclideanSteinerTree
    import numpy as np
    from src.orlib_loader import load_problem_file,load_solution_file

    choice = [1,10,20,30,40,50,60]
    print("Available choices:", choice)
    
    while True:
        try:
            chosen = int(input("Please choose a dataset from the list: "))
            if chosen in choice:
                break
            else:
                print("Invalid choice. Please choose a valid number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    problem_file = f"data/estein{chosen}.txt"
    solution_file = f"data/estein{chosen}opt.txt"

    list_of_problems = load_problem_file(problem_file)
    list_of_solutions = load_solution_file(solution_file)

    max_index = len(list_of_problems) - 1
    
    while True:
        try:
            chosen_index = int(input(f"Please choose an index from 0 to {max_index}: "))
            if 0 <= chosen_index <= max_index:
                break
            else:
                print(f"Invalid choice. Please choose a number between 0 and {max_index}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    

    terminals = np.array(list_of_problems[chosen_index],dtype=np.float32)
    print("Best achievable is :",list_of_solutions[chosen_index]['optimal_value'])
    solution_tree = EuclideanSteinerTree(terminals)
    solution_tree.manually_load_steiner_points(
        np.array(list_of_solutions[chosen_index]['steiner_points'])
    )

    tree = EuclideanSteinerTree(terminals)
    play_interactive_game(tree,solution_tree)
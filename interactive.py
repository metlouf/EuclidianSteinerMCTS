import matplotlib.pyplot as plt

def play_interactive_game(tree):
    """
    Play the tree game interactively in the terminal.
    
    Args:
        tree: Your tree game object with methods like legal_moves(), play_move(), and plot_tree()
    """
    print("Welcome to the Interactive Steiner Tree Game!")
    print("-----------------------------------")
    
    game_over = False


    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Display current state
    print("\nCurrent Tree :")
    tree.fill_plot(ax)
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

    terminals = np.array([(0, 0),(0, 1), (1, 0), (1, 1),(0.5, 0.5)],dtype=np.float32)
    tree = EuclideanSteinerTree(terminals)
    play_interactive_game(tree)
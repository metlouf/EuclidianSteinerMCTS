from itertools import product

def generate_combinations(sets_list, length):
    """
    Generate all combinations of specified length where elements 
    come from different sets.
    
    :param sets_list: List of sets to combine elements from
    :param length: Length of combinations (2 for pairs, 3 for triplets, etc.)
    :return: List of tuples/combinations
    """
    # Check if there are enough sets to create combinations
    if len(sets_list) < length:
        raise ValueError(f"Not enough sets to create {length}-element combinations")
    
    # Use itertools.product to generate combinations
    # Each iteration selects one element from a different set
    combinations = []
    for combo in product(*sets_list[:length]):
        # Ensure all elements are from different sets
        if len(set(combo)) == length:
            combinations.append(combo)
    
    return combinations

# Example usage
def main():
    # Example sets
    sets_list = [
        {1, 2, 3},     # Set 1
        {'a', 'b', 'c'},  # Set 2
        {10, 20, 30},   # Set 3
        {'x', 'y', 'z'},
        {'e', 'f', 'g'}    # Set 4
    ]
    
    # Generate pairs (tuples)
    print("Pairs:")
    pairs = generate_combinations(sets_list, 2)
    for pair in pairs:
        print(pair)
    
    print("\nTriplets:")
    triplets = generate_combinations(sets_list, 3)
    for triplet in triplets:
        print(triplet)
    
    print("\nQuads:")
    quads = generate_combinations(sets_list, 4)
    for quad in quads:
        print(quad)

# Uncomment to run
if __name__ == "__main__":
     main()
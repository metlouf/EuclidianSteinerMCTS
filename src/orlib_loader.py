import numpy as np
import matplotlib.pyplot as plt
from src.steiner import EuclideanSteinerTree

def load_problem_file(filename):
    """
    Load the problem file containing test problems with points.
    
    Returns:
    - List of problems, where each problem is a list of points (x, y coordinates)
    """
    problems = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        line_index = 0
        num_problems = int(lines[line_index].strip())
        line_index += 1
        
        for _ in range(num_problems):
            num_points = int(lines[line_index].strip())
            line_index += 1
            
            points = []
            for j in range(num_points):
                coords = lines[line_index].strip().split()
                x, y = float(coords[0]), float(coords[1])
                points.append((x, y))
                line_index += 1
            
            problems.append(points)
    
    return problems

def load_solution_file(filename):
    """
    Load the solution file containing optimal Steiner solutions.
    
    Returns:
    - List of solutions, where each solution contains:
      * optimal_value: optimal Steiner solution value
      * mst_value: minimal spanning tree solution value
      * steiner_points: list of Steiner points (x, y coordinates)
    """
    solutions = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        line_index = 0
        num_problems = int(lines[line_index].strip())
        line_index += 1
        
        for _ in range(num_problems):
            optimal_value = float(lines[line_index].strip())
            line_index += 1
            
            mst_value = float(lines[line_index].strip())
            line_index += 1
            
            num_steiner_points = int(lines[line_index].strip())
            line_index += 1
            
            steiner_points = []
            for j in range(num_steiner_points):
                coords = lines[line_index].strip().split()
                x, y = float(coords[0]), float(coords[1])
                steiner_points.append((x, y))
                line_index += 1
            
            solutions.append({
                'optimal_value': optimal_value,
                'mst_value': mst_value,
                'steiner_points': steiner_points
            })
    
    return solutions

"""
def main():
    # File paths
    problem_file = "data/estein10.txt"
    solution_file = "data/estein10opt.txt"
    
    # Load problems and solutions
    problems = load_problem_file(problem_file)
    solutions = load_solution_file(solution_file)
    
    print(f"Loaded {len(problems)} problems and {len(solutions)} solutions")
    
    # Print summary
    for i, (problem, solution) in enumerate(zip(problems, solutions)):
        print(f"Problem {i+1}: {len(problem)} points, {len(solution['steiner_points'])} Steiner points")
        print(f"  Optimal value: {solution['optimal_value']:.6f}")
        print(f"  MST value: {solution['mst_value']:.6f}")
        print(f"  Improvement: {(solution['mst_value'] - solution['optimal_value']) / solution['mst_value'] * 100:.2f}%")
        tree = EuclideanSteinerTree(np.array(problem,np.float32))
        print("CHECK MST",tree.get_score())
        tree.manually_load_steiner_points(np.array(solution['steiner_points'],np.float32))
        print("Best Score",tree.get_score())
        tree.plot_tree()
        print()


if __name__ == "__main__":
    main()

"""
# EuclidianSteinerMCTS 

IASD Master Project by : Théo Courty and Mohamed Ali SRIR

## Setup Instructions

Before using this code, you must download the necessary datasets:

```bash
bash download_dataset.sh
```

This script must be run before launching any simulations to ensure all required data is available.

## Dependencies

Install the required Python packages:

```bash
pip install numpy networkx scipy matplotlib
```

The project depends on the following Python libraries:
- NumPy: For numerical operations
- NetworkX: For graph operations and structures
- SciPy: For spatial calculations and optimization
- Matplotlib: For visualization
- Python's standard libraries (itertools, random)

## Project Structure

- `src/steiner.py`: Contains the general structure of the game
- **`interactive.py`**: Provides an interactive interface to illustrate the gameplay
- `naive_methods.py`: Contains implementation of basic MCTS methods
- `uct.py`: Contains implementation of UCT-based MCTS methods
- `orlib_loader.py`: Contains the dataset loader
- `*_xp.sh`: Shell scripts used to obtain the results reported in the tables

## Interactive Demo

The interactive.py script provides a visual interface to understand the gameplay mechanics:

![Demo Animation](demo.gif)

## Branch Information

The `main` branch is up-to-date with the latest changes. Other branches may be deprecated and should not be considered for use.

## Running Experiments

To reproduce the results reported in the paper, run the appropriate experiment script:

```bash
bash <experiment_name>_xp.sh
```

Where `<experiment_name>` corresponds to the specific experiment you wish to run.
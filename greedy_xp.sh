#!/bin/bash

# Start scripts in parallel and capture their PIDs
python3 ./naive_methods.py --chosen 20 --chosen_index 0 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 1 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 2 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 3 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 4 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 5 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 6 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 7 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 8 --mode greedy --parallel

python3 ./naive_methods.py --chosen 20 --chosen_index 9 --mode greedy --parallel

echo "All scripts completed."

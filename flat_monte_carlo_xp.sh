#!/bin/bash

# Start scripts in parallel and capture their PIDs
python naive_methods.py --chosen 20 --chosen_index 0 --mode montecarlo --parallel &
pid0=$!

python naive_methods.py --chosen 20 --chosen_index 1 --mode montecarlo --parallel &
pid1=$!

python naive_methods.py --chosen 20 --chosen_index 2 --mode montecarlo --parallel &
pid2=$!

python naive_methods.py --chosen 20 --chosen_index 3 --mode montecarlo --parallel &
pid3=$!

python naive_methods.py --chosen 20 --chosen_index 4 --mode montecarlo --parallel &
pid4=$!

python naive_methods.py --chosen 20 --chosen_index 5 --mode montecarlo --parallel &
pid5=$!

python naive_methods.py --chosen 20 --chosen_index 6 --mode montecarlo --parallel &
pid6=$!

python naive_methods.py --chosen 20 --chosen_index 7 --mode montecarlo --parallel &
pid7=$!

python naive_methods.py --chosen 20 --chosen_index 8 --mode montecarlo --parallel &
pid8=$!

python naive_methods.py --chosen 20 --chosen_index 9 --mode montecarlo --parallel &
pid9=$!

# Output the PIDs
echo "Started script1.sh with PID $pid0"
echo "Started script1.sh with PID $pid1"
echo "Started script2.sh with PID $pid2"
echo "Started script3.sh with PID $pid3"
echo "Started script1.sh with PID $pid4"
echo "Started script2.sh with PID $pid5"
echo "Started script3.sh with PID $pid6"
echo "Started script1.sh with PID $pid7"
echo "Started script2.sh with PID $pid8"
echo "Started script3.sh with PID $pid9"

# Optionally: Wait for all to finish
wait $pid0
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
wait $pid7
wait $pid8
wait $pid9


echo "All scripts completed."

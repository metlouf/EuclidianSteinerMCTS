#!/bin/bash

# Start scripts in parallel and capture their PIDs
python uct.py --chosen 10 --chosen_index 0  --num_sim 100000 &
pid0=$!

python uct.py --chosen 10 --chosen_index 1  --num_sim 100000 &
pid1=$!

python uct.py --chosen 10 --chosen_index 2  --num_sim 100000 &
pid2=$!

python uct.py --chosen 10 --chosen_index 3  --num_sim 100000 &
pid3=$!

python uct.py --chosen 10 --chosen_index 4  --num_sim 100000 &
pid4=$!

python uct.py --chosen 10 --chosen_index 5  --num_sim 100000 &
pid5=$!

python uct.py --chosen 10 --chosen_index 6  --num_sim 100000 &
pid6=$!

python uct.py --chosen 10 --chosen_index 7  --num_sim 100000 &
pid7=$!

python uct.py --chosen 10 --chosen_index 8  --num_sim 100000 &
pid8=$!

python uct.py --chosen 10 --chosen_index 9  --num_sim 100000 &
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

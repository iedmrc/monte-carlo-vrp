#!/bin/bash
for i in 0 3 5 6 7 8 9
do
tmux new-session -d -s m$i "python3 vrp.py -i 'input_thu.json' -e 5000 -p 100000 -g ${i}"
done

for i in 1 4 5 6 7 8 9
do
tmux new-session -d -s n$i "python3 vrp.py -i 'input_thu.json' -e 5000000 -g ${i}"
done
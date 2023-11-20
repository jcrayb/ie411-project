#!/bin/bash
unset times_full, times_opt, TIME

echo "Full problem"
for ((i=0; i<5; ++i)); do
    TIME=$(python3 primal.py|tail -1)
    echo "Time: $TIME"
    times_full+=($TIME)
done

echo "Optimized problem"
for ((i=0; i<5; ++i)); do
    TIME=$(python3 optimized.py|tail -1)
    echo "Time: $TIME"
    times_opt+=($TIME)
done

echo "Full problem: ${times_full[@]}"
echo "Optimized problem ${times_opt[@]}"
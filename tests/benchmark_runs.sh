#!/bin/bash
# Example:
# bash benchmark_runs.sh benchmark.py 10

# Check that two arguments are passed
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 'script_name' 'nruns'"
    exit 1
fi

total_time=0

# Get the script name and number of runs from command-line arguments
script_name=$1
n_runs=$2

for i in $(seq 1 $n_runs)
do
  output=$(python3 $script_name)

  # Extract the "total time" line and get the time value
  time_value=$(echo "$output" | grep "Total time" | awk '{print $3}')
  # Some scripts use sc.tic() toc() that prints elapsed time
  if [ -z "$time_value" ]; then
    time_value=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
  fi
  total_time=$(awk -v total_time="$total_time" -v time_value="$time_value" 'BEGIN{print total_time + time_value}')
done

# Calculate the average time
average_time=$(awk -v total_time="$total_time" -v n_runs="$n_runs" 'BEGIN{print total_time / n_runs}')

echo "Average runtime (over $n_runs runs): $average_time s"
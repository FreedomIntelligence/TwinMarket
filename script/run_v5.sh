#!/bin/bash

# export http_proxy=http://174.0.250.13:3128
# export https_proxy=http://174.0.250.13:3128

# Create logs directory if it doesn't exist
model='wmh'
log_dir="logs_100_0130_${model}"

mkdir -p "$log_dir"

# Get current timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

# cp data/UserDB/sys_100_hs300.db "${log_dir}/user_100.db"

# Run simulation with nohup
# python simulation.py \
nohup python simulation_ablation_v5.py \
    --log_dir $log_dir \
    --forum_db "${log_dir}/forum_100.db" \
    --user_db "${log_dir}/user_100.db" \
    --user_graph_save_name "user_graph_${log_dir}" \
    --start_date "2023-01-04" \
    --max_workers 50 \
    > "${log_dir}/simulation_${timestamp}.log" 2>&1 &

echo "Simulation started with PID: $!"
echo "Log file: ${log_file}"
tail -f "${log_file}"
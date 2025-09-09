#!/bin/bash

# export http_proxy=http://174.0.250.13:3128
# export https_proxy=http://174.0.250.13:3128

# Create logs directory if it doesn't exist
model='4o-mini_w_news'
log_dir="logs_100_0129_${model}"

mkdir -p "$log_dir"

# Get current timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

# cp data/UserDB/sys_100.db "${log_dir}/user_100.db"

# Run simulation with nohup
# python simulation.py \
nohup python simulation_ablation_v4.py \
    --log_dir $log_dir \
    --forum_db "${log_dir}/forum_100.db" \
    --user_db "${log_dir}/user_100.db" \
    --user_graph_save_name "user_graph_${log_dir}" \
    --start_date "2023-06-15" \
    --end_date "2023-07-15" \
    --max_workers 100 \
    --top_n_user 1 \
    > "${log_dir}/simulation_${timestamp}.log" 2>&1 &

pid=$!
echo "Simulation started with PID: $pid"
echo "Log file: ${log_file}"
echo "Process started with PID: $pid" >> "${log_file}"
tail -f "${log_file}"
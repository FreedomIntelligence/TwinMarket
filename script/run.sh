#!/bin/bash

# export http_proxy=http://174.0.250.13:3128
# export https_proxy=http://174.0.250.13:3128

# Create logs directory if it doesn't exist
log_dir="logs_1000"
mkdir -p "$log_dir"

# Get current timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

# Run simulation with nohup
# python simulation.py \
nohup python simulation_v3.py \
    --start_date "2023-06-19" \
    --end_date "2023-12-15" \
    --forum_db "data/ForumDB/sys_1000.db" \
    --user_db "data/UserDB/sys_1000.db" \
    --debug true \
    --max_workers 300 \
    --node 1000 \
    --user_graph_save_name "user_1000" \
    --similarity_threshold 0.15 \
    --time_decay_factor 0.05 \
    --checkpoint false \
    --log_dir $log_dir \
    --checkpoint true \
    --prob_of_technical 0.5 \
    > "${log_dir}/simulation_${timestamp}.log" 2>&1 &

# Print process ID and log file location
echo "Simulation started with PID: $!"
echo "Log file: ${log_file}"

# Follow the log file
tail -f "${log_file}"
#!/bin/bash

export http_proxy=http://174.0.250.13:3128
export https_proxy=http://174.0.250.13:3128

# Create logs directory if it doesn't exist
length=100
model='rebuttal_gemini_0327_2'
log_dir="logs_${length}_${model}"

mkdir -p "$log_dir"

# Get current timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

# cp data/UserDB/sys_100.db "${log_dir}/user_100.db"

# Run simulation with nohup
# python simulation.py \
nohup python simulation_ablation_v4.py \
    --log_dir $log_dir \
    --forum_db "${log_dir}/forum_${length}.db" \
    --user_db "${log_dir}/user_${length}.db" \
    --user_graph_save_name "user_graph_${log_dir}" \
    --start_date "2023-6-28" \
    --end_date "2023-8-15" \
    --max_workers 50 \
    --config_path './config_random/gemini.yaml' \
    > "${log_dir}/simulation_${timestamp}.log" 2>&1 &

echo "Simulation started with PID: $!"
echo "Log file: ${log_file}"
tail -f "${log_file}"
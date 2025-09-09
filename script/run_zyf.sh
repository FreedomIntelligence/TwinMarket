#!/bin/bash

# export http_proxy=http://174.0.250.13:3128
# export https_proxy=http://174.0.250.13:3128

# Create logs directory if it doesn't exist
log_dir="logs_100_ablation_all"
mkdir -p "$log_dir"

# Get current timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

export PYTHONPATH=$PYTHONPATH:/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket

# Run simulation with nohup
# python simulation.py \
python simulation_ablation_v3.py \
    --start_date "2023-06-15" \
    --end_date "2023-12-15" \
    --forum_db "data/ForumDB/sys_100_ablation_all.db" \
    --user_db "data/UserDB/sys_100_ablation_all.db" \
    --debug true \
    --max_workers 100 \
    --node 100 \
    --user_graph_save_name "user_graph_ablation_100" \
    --similarity_threshold 0.1 \
    --time_decay_factor 0.05 \
    --checkpoint false \
    --log_dir $log_dir \
    --prob_of_technical 0.5 \
    --belief_init_path "util/belief/belief_100.csv"
    # > "${log_dir}/simulation_${timestamp}.log" 2>&1 &

# Print process ID and log file location
echo "Simulation started with PID: $!"
echo "Log file: ${log_file}"

# Follow the log file
tail -f "${log_file}"
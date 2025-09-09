#!/bin/bash

export http_proxy=http://174.0.250.13:3128
export https_proxy=http://174.0.250.13:3128

length=1000
model='gemini'
activate_prob="0.8"
log_dir="logs_${length}_nips_${model}_${activate_prob}"

mkdir -p "$log_dir"

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

# Run simulation with nohup
nohup python simulation_ablation_v4_nips_activate.py \
    --log_dir $log_dir \
    --forum_db "${log_dir}/forum_${length}.db" \
    --user_db "${log_dir}/user_${length}.db" \
    --user_graph_save_name "user_graph_${log_dir}" \
    --start_date "2023-7-15" \
    --end_date "2023-8-15" \
    --max_workers 250 \
    --config_path './config_random/new.yaml' \
    --activate_prob ${activate_prob} \
    > "${log_file}" 2>&1 &

echo "Simulation started with PID: $!" >> "${log_file}"
echo "Simulation started with PID: $!"
echo "Log file: ${log_file}"
tail -f "${log_file}"
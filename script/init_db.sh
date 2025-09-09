#!/bin/bash

log_dir="../logs_1000"
mkdir -p "$log_dir"
mkdir -p "$log_dir/db_logs"  

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/db_logs/init_db_${timestamp}.log"

# 设置参数
n_users=1000
start_date="2023-06-15"

nohup python ../data/UserDB/init_db.py \
    --n_users ${n_users} \
    --start_date "${start_date}" \
    > "${log_file}" 2>&1 &

# 获取进程ID
pid=$!

echo "数据库初始化已启动，PID: $pid"
echo "日志文件: ${log_file}"

while kill -0 $pid 2>/dev/null; do
    tail -f "${log_file}" & tail_pid=$!
    sleep 1
    if ! kill -0 $pid 2>/dev/null; then
        kill $tail_pid 2>/dev/null
        echo "初始化完成！"
        break
    fi
done

sleep 1
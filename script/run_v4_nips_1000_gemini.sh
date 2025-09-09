#!/bin/bash

length=1000
model='gemini'
activate_prob="0.8"
log_dir="logs_${length}_nips_${model}_${activate_prob}"

rm -rf $log_dir
# 创建日志目录
mkdir -p "$log_dir"

# 检查并自动设置数据库文件
user_db_path="${log_dir}/user_${length}.db"
forum_db_path="${log_dir}/forum_${length}.db"

echo "=== 自动数据库设置 ==="

# 检查用户数据库
if [[ ! -f "$user_db_path" || $(stat -f%z "$user_db_path" 2>/dev/null || stat -c%s "$user_db_path" 2>/dev/null) -eq 0 ]]; then
    echo "用户数据库不存在或为空，从模板复制..."
    if [[ -f "data/UserDB/sys_${length}.db" ]]; then
        cp "data/UserDB/sys_${length}.db" "$user_db_path"
        echo "✅ 复制用户数据库: sys_${length}.db → user_${length}.db"
    else
        echo "❌ 找不到模板数据库: data/UserDB/sys_${length}.db"
        exit 1
    fi
else
    echo "✅ 用户数据库已存在: $user_db_path"
fi

# 检查论坛数据库
if [[ ! -f "$forum_db_path" || $(stat -f%z "$forum_db_path" 2>/dev/null || stat -c%s "$forum_db_path" 2>/dev/null) -eq 0 ]]; then
    echo "论坛数据库不存在或为空，立即初始化..."
    rm -f "$forum_db_path"  # 删除空文件
    python3 -c "
from util.ForumDB import init_db_forum
print('正在初始化Forum数据库...')
init_db_forum(db_path='$forum_db_path')
print('✅ Forum数据库初始化完成!')
"
    if [[ $? -eq 0 ]]; then
        echo "✅ 论坛数据库初始化成功: $forum_db_path"
    else
        echo "❌ 论坛数据库初始化失败"
        exit 1
    fi
else
    echo "✅ 论坛数据库已存在: $forum_db_path"
fi

echo "===================="

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/simulation_${timestamp}.log"

# Run simulation (同时输出到终端和日志)
echo "Simulation starting..."
echo "Log file: ${log_file}"
echo "===================="

python simulation.py \
    --log_dir $log_dir \
    --forum_db "${log_dir}/forum_${length}.db" \
    --user_db "${log_dir}/user_${length}.db" \
    --user_graph_save_name "user_graph_${log_dir}" \
    --start_date "2023-06-15" \
    --end_date "2023-8-15" \
    --max_workers 250 \
    --config_path './config/api.yaml' \
    --activate_prob ${activate_prob} \
    2>&1 | tee "${log_file}"

echo "Simulation completed!"
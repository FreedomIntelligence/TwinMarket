

import pandas as pd
import math
import random
from datetime import datetime
import json
from openai import OpenAI
from datetime import datetime, timedelta
import argparse
from typing import Dict, Literal, Optional
import os
import yaml
import trader.TradingAgentV2 as TradingAgent
from util.UserDB import get_all_user_ids, get_user_profile, build_graph, load_graph, update_graph, save_graph, build_graph_new, get_top_n_users_by_degree
from util.ForumDB import init_db_forum, execute_forum_actions, update_posts_score_by_date, update_posts_score_by_date_range, create_post_db, get_all_users_posts_db
import sqlite3
from tqdm import tqdm
import asyncio
import logging
from trader.broker import test_matching_system, update_profiles_table_holiday
from trader.utility import init_system
from Agent import BaseAgent
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import closing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_user_input(user_id, user_db, forum_db, df_stock, current_date, debug, day_1st, current_user_graph, import_news, df_strategy, is_trading_day, top_user, log_dir, prob_of_technical, user_config_mapping, activate_maapping, belief_args):
    try:
        # 获取用户策略
        user_strategy = df_strategy[df_strategy['user_id'] == user_id].iloc[0]['strategy']
        is_random_trader = user_strategy == "技术面" and user_id not in top_user and is_trading_day and random.random() < prob_of_technical

        previous_date = current_date - timedelta(days=1)
        previous_date_str = previous_date.strftime('%Y-%m-%d 00:00:00')
        user_profile = get_user_profile(db_path=user_db, user_id=user_id, created_at=previous_date_str)

        stock_ids = list(user_profile["cur_positions"].keys()) if user_profile.get("cur_positions") else []
        is_top_user = user_id in top_user
        is_activate_user = activate_maapping[user_id]
        
        if belief_args is None:
            belief = belief_args.get(user_id)[0]['belief']
        else:
            belief = None

        tradingAgent = TradingAgent.PersonalizedStockTrader(
            user_profile=user_profile,
            user_graph=current_user_graph,
            forum_db_path=forum_db,
            user_db_path=user_db,
            df_stock=df_stock,
            import_news=import_news,
            user_strategy=user_strategy,
            is_trading_day=is_trading_day,
            is_top_user=is_top_user,
            log_dir=log_dir,
            is_random_trader=is_random_trader,
            config_path=user_config_mapping[user_id],
            is_activate_user=is_activate_user,
            belief=belief
        )

        forum_args, user_id, decision_result, post_response_args, conversation_history = tradingAgent.input_info(
            stock_codes=stock_ids,
            current_date=current_date,
            debug=debug,
            day_1st=day_1st
        )
        
        # log conversation history
        if conversation_history:
            conversation_dir = os.path.join(f"{log_dir}/conversation_records/{current_date.strftime('%Y-%m-%d')}")
            os.makedirs(conversation_dir, exist_ok=True)
            conversation_file = os.path.join(conversation_dir, f"{user_id}.json")
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=4, ensure_ascii=False)

        return user_id, forum_args, decision_result, post_response_args
    except Exception as e:
        logging.error(f"Error processing user {user_id}: {e}")
        return user_id, {"error": str(e)}, None, None


def init_simulation(
    start_date: pd.Timestamp = pd.Timestamp('2023-06-15'),
    end_date: pd.Timestamp = pd.Timestamp('2023-06-16'),
    forum_db: str = 'data/ForumDB/sample.db',
    user_db: str = 'data/UserDB/sys_100.db',
    debug: bool = True,
    max_workers: int = 1,
    user_graph_save_name: str = 'user_graph',
    checkpoint: bool = True,
    similarity_threshold: float = 0.1,
    time_decay_factor: float = 0.05,
    node: int = 1000,
    log_dir: str = 'logs',
    prob_of_technical: float = 0.3
):
    """
    初始化模拟交易
    """
    current_date = start_date

    # 清空未来的数据库
    init_system(current_date, user_db, forum_db)

    # 读取全体重要新闻广播
    df_news = pd.read_pickle('data/update_long_news/sorted_impact_news.pkl')
    df_news['cal_date'] = pd.to_datetime(df_news['cal_date'])

    # 获取所有交易日
    df_trading_days = pd.read_csv('data/test_agent_zyf/basic_data/trading_days.csv')
    df_trading_days['pretrade_date'] = pd.to_datetime(df_trading_days['pretrade_date'])
    trading_days = list(df_trading_days['pretrade_date'].unique())

    # 读取用户类型
    conn = sqlite3.connect(user_db)
    df_strategy = pd.read_sql_query("SELECT * FROM Strategy;", conn)
    df_strategy['user_id'] = df_strategy['user_id'].astype(str)
    conn.close()

    while current_date <= end_date:

        if checkpoint:
            day_1st = False
        else:
            day_1st = (current_date == start_date)

        # 判断当天是否是交易日
        is_trading_day = (current_date in trading_days)

        # 获取当天对应的股票信息
        if is_trading_day:
            conn = sqlite3.connect(user_db)
            df_stock = pd.read_sql_query("SELECT * FROM StockData;", conn)
            df_stock['date'] = pd.to_datetime(df_stock['date'])
            conn.close()
        else:
            df_stock = None

        # 获取当天对应的新闻
        import_news = df_news[df_news['cal_date'] == current_date].iloc[0]['news']

        print(f"\n=== Current Date: {current_date.strftime('%Y-%m-%d')} ===")
        print(f"Trading Day: {is_trading_day}")
        all_user = get_all_user_ids(db_path=user_db,
                                    timestamp=current_date)

        config_list = ['./config_random/deepseek_yyz.yaml',
                       './config_random/deepseek_yyz2.yaml',
                       './config_random/deepseek_yyz3.yaml',
                       './config_random/deepseek_zyf1.yaml',
                       './config_random/deepseek_zyf2.yaml',
                       './config_random/deepseek_zyf3.yaml',
                       './config_random/deepseek_wmh.yaml',
                       './config_random/deepseek_wmh_2.yaml',
                       #    './config_random/gaochao_4o.yaml',
                       #    './config_random/gaochao_4o_mini.yaml'
                       ]
        config_prob = [0.2, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.15]  # todo
        # config_prob = [1, 0, 0, 0]
        user_config_mapping = {}
        for user_id in all_user:
            # 随机选择一个配置文件路径
            selected_config = random.choices(config_list, weights=config_prob, k=1)[0]
            user_config_mapping[user_id] = selected_config

        activate_maapping = {}
        activate_agent_prob = 0.3
        for user_id in all_user:
            activate = random.random() < activate_agent_prob
            activate_maapping[user_id] = activate
            
        belief_args = {}
        if not day_1st:
            belief_args = get_all_users_posts_db(db_path=forum_db, end_date=current_date)

        current_user_graph = build_graph_new(
            similarity_threshold=similarity_threshold,
            time_decay_factor=time_decay_factor,
            db_path=user_db,
            start_date='2023-1-1',
            end_date=(current_date-timedelta(1)).strftime('%Y-%m-%d'),
            save_name=f'{user_graph_save_name}_{current_date.strftime("%Y-%m-%d")}',
            save=True
        )
        print(f"Graph properties: {current_user_graph.number_of_nodes()} nodes, {current_user_graph.number_of_edges()} edges.")
        top_user = get_top_n_users_by_degree(G=current_user_graph, top_n=int(node*0.1))

        # 使用 ThreadPoolExecutor 并发处理用户
        print(f"Processing {len(all_user)} users with {max_workers} workers...")
        results = []
        forum_args_list = []
        post_args_list = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_user_input, user_id, user_db, forum_db, df_stock, current_date, debug, day_1st, current_user_graph, import_news, df_strategy, is_trading_day, top_user, log_dir, prob_of_technical, user_config_mapping, activate_maapping, belief_args
                )
                for user_id in all_user
            ]

            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(futures), total=len(all_user), desc=f"INPUT {current_date.strftime('%Y-%m-%d')}", unit="user"):
                try:
                    user_id, forum_args, decision_result, post_response_args = future.result()  # 等待每个线程完成并获取结果
                    # results.append((user_id, result))
                    forum_args_list.append((user_id, forum_args))
                    results.append((user_id, decision_result))
                    post_args_list.append((user_id, post_response_args))
                except Exception as e:
                    print(f"[INPUT] Error processing user: {e}")

        # 将当天所有用户的结果保存到 JSON 文件
        result_dir = os.path.join(f"{log_dir}/trading_records")
        os.makedirs(result_dir, exist_ok=True)  # 确保结果目录存在
        result_file = os.path.join(result_dir, f"{current_date.strftime('%Y-%m-%d')}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            result_dict = {user_id: result for user_id, result in results}
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        reacion_result_dir = os.path.join(f"{log_dir}/reaction_records")
        os.makedirs(reacion_result_dir, exist_ok=True)
        reaction_result_file = os.path.join(reacion_result_dir, f"{current_date.strftime('%Y-%m-%d')}.json")
        with open(reaction_result_file, "w", encoding="utf-8") as f:
            reaction_result_dict = {user_id: reaction_result for user_id, reaction_result in forum_args_list}
            json.dump(reaction_result_dict, f, indent=4, ensure_ascii=False)

        post_result_dir = os.path.join(f"{log_dir}/post_records")
        os.makedirs(post_result_dir, exist_ok=True)
        post_result_file = os.path.join(post_result_dir, f"{current_date.strftime('%Y-%m-%d')}.json")
        with open(post_result_file, "w", encoding="utf-8") as f:
            post_result_dict = {user_id: post_result for user_id, post_result in post_args_list}
            json.dump(post_result_dict, f, indent=4, ensure_ascii=False)

        if post_args_list:
            for user_id, post_response_args in post_args_list:
                try:
                    create_post_db(
                        user_id=user_id,
                        content=post_response_args["post"],
                        type=post_response_args["type"],
                        belief=str(post_response_args["belief"]),
                        created_at=current_date,
                        db_path=forum_db)
                except Exception as e:
                    print(f"[POST ACTION] Error processing forum actions for user {user_id}: {e}")
                    print(post_response_args)

        # 根据是否是交易日选择更新的函数
        if is_trading_day:
            test_matching_system(
                current_date=current_date.strftime('%Y-%m-%d'),
                base_path=log_dir,
                db_path=user_db,
                json_file_path=f"{log_dir}/trading_records/{current_date.strftime('%Y-%m-%d')}.json"
            )
        if not is_trading_day:
            update_profiles_table_holiday(current_date=current_date.strftime('%Y-%m-%d'),
                                          db_path=user_db)

        if not day_1st:
            # 统一处理 forum_args 并写入数据库
            print(f"Processing forum actions for {len(forum_args_list)} users...")
            if forum_args_list:
                for user_id, forum_args in forum_args_list:
                    try:
                        asyncio.run(execute_forum_actions(
                            forum_args=forum_args,
                            db_path=forum_db,
                            user_id=user_id,
                            created_at=current_date
                        ))
                    except Exception as e:
                        print(f"[FORUM ACTION] Error processing forum actions for user {user_id}: {e}")
            update_posts_score_by_date_range(db_path=forum_db,
                                             end_date=current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize stock trading simulation.")
    parser.add_argument("--start_date", type=str, default="2023-06-15", help="Start date of the simulation (format: YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2023-07-15", help="End date of the simulation (format: YYYY-MM-DD).")
    parser.add_argument("--forum_db", type=str, default="data/ForumDB/sys_10.db", help="Path to the forum database.")
    parser.add_argument("--user_db", type=str, default="data/UserDB/sys_10.db", help="Path to the user database.")
    parser.add_argument("--debug", type=bool, default=True, help="Enable debug mode.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads for concurrent processing.")
    parser.add_argument("--user_graph_save_name", type=str, default="user_graph", help="Name of the user graph file.")
    parser.add_argument("--checkpoint", type=bool, default=False, help="Start from checkpoint.")
    parser.add_argument("--similarity_threshold", type=float, default=0.1, help="Similarity threshold for building user graph.")
    parser.add_argument("--time_decay_factor", type=float, default=0.05, help="Time decay factor for building user graph.")
    parser.add_argument("--node", type=int, default=10, help="Number of nodes in the user graph.")
    parser.add_argument("--log_dir", type=str, default="logs_sample", help="Directory to save log files.")
    parser.add_argument("--prob_of_technical", type=float, default=0.5, help="Probability of technical noise trader.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # 判断log_dir是否存在
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.checkpoint == False:
        print("创建forumDB")
        init_db_forum(db_path=args.forum_db)

    print(json.dumps(vars(args), indent=4, ensure_ascii=False))

    # 运行模拟
    init_simulation(
        start_date=pd.Timestamp(args.start_date),
        end_date=pd.Timestamp(args.end_date),
        forum_db=args.forum_db,
        user_db=args.user_db,
        debug=args.debug,
        max_workers=args.max_workers,
        user_graph_save_name=args.user_graph_save_name,
        checkpoint=args.checkpoint,
        similarity_threshold=args.similarity_threshold,
        time_decay_factor=args.time_decay_factor,
        node=args.node,
        log_dir=args.log_dir,
        prob_of_technical=args.prob_of_technical
    )

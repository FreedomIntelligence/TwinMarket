"""
交易模拟系统主程序


该模块负责运行完整的股票交易模拟，包括用户行为、论坛互动、交易执行等。
"""

# 标准库导入
import argparse
import asyncio
import json
import logging
import math
import os
import random
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta
from typing import Dict, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# 第三方库导入
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

# 本地模块导入
import trader.trading_agent as TradingAgent
from Agent import BaseAgent
from trader.matching_engine import test_matching_system, update_profiles_table_holiday
from trader.utility import init_system
from util.UserDB import (
    get_all_user_ids,
    get_user_profile,
    build_graph,
    load_graph,
    update_graph,
    save_graph,
    build_graph_new,
    get_top_n_users_by_degree,
)
from util.ForumDB import (
    init_db_forum,
    execute_forum_actions,
    update_posts_score_by_date,
    update_posts_score_by_date_range,
    create_post_db,
    get_all_users_posts_db,
)

# ============================ 全局配置常量 ============================
# 代理激活概率（默认全部激活）
ACTIVATE_AGENT_PROB = 1

# 超时阈值（秒）- 5分钟
TIMEOUT_THRESHOLD = 5 * 60

# 配置日志输出格式
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 线程锁，用于保护并发文件写入操作
lock = threading.Lock()


def process_user_input(
    user_id,
    user_db,
    forum_db,
    df_stock,
    current_date,
    debug,
    day_1st,
    current_user_graph,
    import_news,
    df_strategy,
    is_trading_day,
    top_user,
    log_dir,
    prob_of_technical,
    user_config_mapping,
    activate_maapping,
    belief_args,
    config_path,
):
    """
    处理单个用户的交易输入和决策过程

    Args:
        user_id: 用户ID
        user_db: 用户数据库路径
        forum_db: 论坛数据库路径
        df_stock: 股票数据 DataFrame
        current_date: 当前日期
        debug: 是否开启调试模式
        day_1st: 是否为第一天
        current_user_graph: 当前用户关系图
        import_news: 导入的新闻数据
        df_strategy: 用户策略数据
        is_trading_day: 是否为交易日
        top_user: 顶级用户列表
        log_dir: 日志目录
        prob_of_technical: 技术面交易者概率
        user_config_mapping: 用户配置映射
        activate_maapping: 用户激活映射
        belief_args: 信念值参数
        config_path: 配置文件路径

    Returns:
        tuple: (user_id, forum_args, decision_result, post_response_args)
    """
    try:
        # 获取用户交易策略
        user_strategy = df_strategy[df_strategy["user_id"] == user_id].iloc[0][
            "strategy"
        ]
        # 判断是否为随机技术面交易者（非顶级用户且交易日且符合概率）
        is_random_trader = (
            user_strategy == "技术面"
            and user_id not in top_user
            and is_trading_day
            and random.random() < prob_of_technical
        )

        # 获取用户历史信息（前一天的数据）
        previous_date = current_date - timedelta(days=1)
        previous_date_str = previous_date.strftime("%Y-%m-%d 00:00:00")
        user_profile = get_user_profile(
            db_path=user_db, user_id=user_id, created_at=previous_date_str
        )

        # 获取用户当前持仓股票ID列表
        stock_ids = (
            list(user_profile["cur_positions"].keys())
            if user_profile.get("cur_positions")
            else []
        )
        # 设置用户状态标识
        is_top_user = user_id in top_user
        is_activate_user = activate_maapping[user_id]

        # 获取用户信念值（从不同数据源根据日期获取）
        try:
            belief = None
            if not day_1st and isinstance(belief_args, dict):
                # 非第一天：从论坛数据库获取信念值（可能不存在）
                user_posts = belief_args.get(str(user_id))
                if (
                    isinstance(user_posts, list)
                    and len(user_posts) > 0
                    and isinstance(user_posts[0], dict)
                    and "belief" in user_posts[0]
                ):
                    belief = user_posts[0]["belief"]
            else:
                # 第一天或备用方案：从CSV文件获取初始信念值（可能不存在）
                try:
                    user_id_str = str(user_id)
                    belief_series = belief_args.loc[
                        belief_args["user_id"].astype(str) == user_id_str, "belief"
                    ]
                    belief = belief_series.iloc[0] if not belief_series.empty else None
                except Exception:
                    belief = None
        except Exception as e:
            print(f"获取用户 {user_id} 的信念值时出错: {e}")
            belief = None

        # 提示未能获取到 belief 的情况
        # 仅对已激活用户提示缺少 belief；未激活用户静默
        if belief is None and is_activate_user:
            logging.warning(
                f"[belief] 用户 {user_id} 未能获取到 belief，将以 None 继续"
            )

        # 创建个性化股票交易代理，传入用户相关信息和配置
        tradingAgent = TradingAgent.PersonalizedStockTrader(
            user_profile=user_profile,  # 用户资料和持仓信息
            user_graph=current_user_graph,  # 用户关系网络图
            forum_db_path=forum_db,  # 论坛数据库路径
            user_db_path=user_db,  # 用户数据库路径
            df_stock=df_stock,  # 股票数据
            import_news=import_news,  # 当日新闻信息
            user_strategy=user_strategy,  # 用户交易策略
            is_trading_day=is_trading_day,  # 是否交易日
            is_top_user=is_top_user,  # 是否为顶级用户
            log_dir=log_dir,  # 日志目录
            is_random_trader=is_random_trader,  # 是否为随机交易者
            config_path=config_path,  # API配置文件路径
            is_activate_user=is_activate_user,  # 用户是否激活
            belief=belief,  # 用户信念值
        )

        # 调用交易代理的主要处理逻辑，获取交易决策和论坛互动结果
        (
            forum_args,
            user_id,
            decision_result,
            post_response_args,
            conversation_history,
        ) = tradingAgent.input_info(
            stock_codes=stock_ids,  # 用户持有的股票代码列表
            current_date=current_date,  # 当前日期
            debug=debug,  # 调试模式标识
            day_1st=day_1st,  # 是否为第一天
        )

        # 保存用户与AI的对话记录（线程安全）
        if conversation_history:
            # 创建对话记录目录
            conversation_dir = os.path.join(
                f"{log_dir}/conversation_records/{current_date.strftime('%Y-%m-%d')}"
            )
            os.makedirs(conversation_dir, exist_ok=True)
            conversation_file = os.path.join(conversation_dir, f"{user_id}.json")

            # 使用线程锁保护并发文件写入操作
            with lock:
                with open(conversation_file, "w", encoding="utf-8") as f:
                    json.dump(conversation_history, f, indent=4, ensure_ascii=False)

        # 返回处理结果：用户ID、论坛互动参数、交易决策结果、帖子回复参数
        return user_id, forum_args, decision_result, post_response_args

    except Exception as e:
        # 异常处理：打印错误信息并返回错误状态
        print(f"处理用户 {user_id} 时出错: {e}")
        return user_id, {"error": str(e)}, None, None


def init_simulation(
    start_date: pd.Timestamp = pd.Timestamp("2023-06-15"),
    end_date: pd.Timestamp = pd.Timestamp("2023-06-16"),
    forum_db: str = "data/ForumDB/sample.db",
    user_db: str = "data/UserDB/sys_100.db",
    debug: bool = True,
    max_workers: int = 1,
    user_graph_save_name: str = "user_graph",
    checkpoint: bool = True,
    similarity_threshold: float = 0.1,
    time_decay_factor: float = 0.05,
    node: int = 1000,
    log_dir: str = "logs",
    prob_of_technical: float = 0.3,
    belief_init_path: str = "util/belief/belief_1000_0129.csv",
    top_n_user: float = 0.1,
    config_path: str = "./config/api.yaml",
    activate_prob: float = 1.0,
):
    """
    初始化并运行股票交易模拟系统

    该函数是整个模拟系统的核心，负责初始化所有必要的组件并按日期顺序执行模拟。

    Args:
        start_date: 模拟开始日期
        end_date: 模拟结束日期
        forum_db: 论坛数据库文件路径
        user_db: 用户数据库文件路径
        debug: 是否开启调试模式
        max_workers: 最大并发线程数
        user_graph_save_name: 用户关系图保存名称
        checkpoint: 是否从检查点开始
        similarity_threshold: 用户相似度阈值
        time_decay_factor: 时间衰减因子
        node: 用户节点数量
        log_dir: 日志输出目录
        prob_of_technical: 技术面交易者激活概率
        belief_init_path: 初始信念值文件路径
        top_n_user: 顶级用户比例
        config_path: API配置文件路径
        activate_prob: 用户激活概率
    """
    # ============================ 模拟初始化 ============================
    current_date = start_date

    # 清空数据库中未来日期的数据，确保模拟的一致性
    init_system(current_date, user_db, forum_db)

    # 加载重要新闻数据（已按影响力排序）
    df_news = pd.read_pickle("data/update_long_news/sorted_impact_news.pkl")
    df_news["cal_date"] = pd.to_datetime(df_news["cal_date"])

    # 加载交易日历数据，用于判断当日是否为交易日
    df_trading_days = pd.read_csv("data/test_agent_zyf/basic_data/trading_days.csv")
    df_trading_days["pretrade_date"] = pd.to_datetime(df_trading_days["pretrade_date"])
    trading_days = list(df_trading_days["pretrade_date"].unique())

    # 从数据库加载用户交易策略信息
    conn = sqlite3.connect(user_db)
    df_strategy = pd.read_sql_query("SELECT * FROM Strategy;", conn)
    df_strategy["user_id"] = df_strategy["user_id"].astype(str)
    conn.close()

    # 加载用户初始信念值数据
    df_init_belief = pd.read_csv(belief_init_path)
    df_init_belief["user_id"] = df_init_belief["user_id"].astype(str)

    # ============================ 主模拟循环 ============================
    while current_date <= end_date:

        # 判断是否为第一天（影响数据加载和初始化逻辑）
        if checkpoint:
            day_1st = False  # 从检查点开始，不是第一天
        else:
            day_1st = current_date == start_date  # 正常开始，检查是否为起始日期

        # 检查当前日期是否为交易日
        is_trading_day = current_date in trading_days

        # 根据是否为交易日加载相应的股票数据
        if is_trading_day:
            # 交易日：从数据库加载股票数据
            conn = sqlite3.connect(user_db)
            df_stock = pd.read_sql_query("SELECT * FROM StockData;", conn)
            df_stock["date"] = pd.to_datetime(df_stock["date"])
            conn.close()
        else:
            # 非交易日：不需要股票数据
            df_stock = None

        # 获取当日对应的新闻信息
        import_news = df_news[df_news["cal_date"] == current_date].iloc[0]["news"]
        # # 获取当天对应的新闻
        # if not day_1st:
        #     import_news = df_news[df_news['cal_date'] == current_date].iloc[0]['news']
        # else:
        #     # TLEI
        #     import_news = [
        #         '最新公布的中国制造业采购经理人指数（PMI）数据不仅再次不及预期，更呈现断崖式下跌，跌破荣枯线多个百分点。这不仅证实了制造业复苏动能的彻底丧失，更释放了中国经济可能加速进入衰退的强烈信号。市场担忧情绪蔓延，投资者恐慌抛售，预期经济硬着陆的风险急剧上升。更有分析师表示，当前的PMI数据反映的可能不是简单的复苏乏力，而是经济结构的深层崩溃。',
        #         '受美联储持续加息和全球避险情绪升温影响，美元指数强势上涨，人民币汇率连日暴跌，引发大规模资本外逃潮。市场传言，外资机构正以惊人速度抛售人民币资产，大量资金涌向美元避险，人民币资产价值面临崩溃。更有分析师警告，人民币贬值可能引发恶性循环，进一步加剧国内经济下行压力。',
        #         '美国政府突然宣布对中国进口商品加征惩罚性关税，贸易战不仅再次升级，更可能演变为一场全面的经济对抗。此举将严重冲击中国外贸，导致出口订单锐减，大量企业倒闭，失业率飙升。市场普遍认为，中美贸易战的升级将加速中国经济的衰退进程，经济寒冬可能提前来临。投资者恐慌情绪急剧升温，A股市场抛售潮不断。'
        #         '全球贸易萎缩加剧，航运业遭受史无前例的重创，作为行业龙头的中国远洋海运集团（中远海控SH601919）也面临破产风险。市场传言，公司债务缠身，资产负债表彻底恶化，即将宣布破产重组，股票价值可能归零。此消息一出，整个航运板块哀鸿遍野，恐慌情绪迅速蔓延至整个A股市场，投资者纷纷逃离。'
        #         '受经济下行影响，高端消费市场彻底崩塌，曾经被视为“硬通货”的贵州茅台（SH600519）等高端白酒销售额大幅下滑。市场传言，茅台的经销商体系已经崩溃，库存积压如山，即将被迫降价促销。曾经高不可攀的“茅台神话”彻底破灭，股价可能断崖式下跌，并引发整个消费板块的恐慌性抛售。投资者对中国消费市场彻底失去信心。'
        #     ]

        # 输出当前模拟日期信息
        print(f"\n=== 当前日期: {current_date.strftime('%Y-%m-%d')} ===")
        print(f"交易日: {is_trading_day}")

        # 获取当前日期有效的所有用户ID
        all_user = get_all_user_ids(db_path=user_db, timestamp=current_date)

        # config_list = ['./config_random/deepseek_yyz.yaml',
        #                './config_random/deepseek_yyz2.yaml',
        #                './config_random/deepseek_yyz3.yaml',
        #                './config_random/deepseek_yyz4.yaml',
        #                './config_random/deepseek_yyz5.yaml',
        #                './config_random/deepseek_zyf1.yaml',
        #                './config_random/deepseek_zyf2.yaml',
        #                './config_random/deepseek_zyf3.yaml',
        #                './config_random/deepseek_zyf4.yaml',
        #                './config_random/deepseek_wmh.yaml',
        #                './config_random/deepseek_wmh2.yaml',
        #                './config_random/deepseek_wmh3.yaml',
        #                #    './config_random/gaochao_4o.yaml',
        #                #    './config_random/gaochao_4o_mini.yaml'
        #                ]
        # config_prob = [0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]  # todo

        # ============================ API配置管理 ============================
        # 使用单一配置文件（可扩展为多个API配置的随机选择）
        config_list = [config_path]
        print(f"使用的API配置: {config_list}")

        # 备注：以下为其他可用的API配置选项
        # config_list = ['./config_random/zyf.yaml']                    # 自定义配置
        # config_list = ['./config_random/gemini-2.0-flash-exp.yaml']   # Gemini 2.0
        # config_list = ['./config_random/claude_3.5_sonnet.yaml']      # Claude 3.5
        # config_list = ['./config_random/gemini-1.5-flash_latest.yaml'] # Gemini 1.5

        config_prob = [1]  # 配置概率权重（单个配置时为1）

        # 为每个用户随机分配API配置
        user_config_mapping = {}
        for user_id in all_user:
            # 按权重随机选择配置文件
            selected_config = random.choices(config_list, weights=config_prob, k=1)[0]
            user_config_mapping[user_id] = selected_config

        # ============================ 用户激活状态管理 ============================
        # 为每个用户随机决定是否激活（根据激活概率）
        activate_maapping = {}
        for user_id in all_user:
            # 根据设定的激活概率决定用户是否参与交易
            activate = random.random() < activate_prob
            activate_maapping[user_id] = activate

        # ============================ 用户信念值管理 ============================
        # 根据是否为第一天选择不同的信念值数据源
        belief_args = {}
        if not day_1st:
            # 非第一天：从论坛数据库获取用户最新信念值
            belief_args = get_all_users_posts_db(
                db_path=forum_db, end_date=current_date
            )
            # 统一字典 key 为字符串，避免 int/str 不一致
            if isinstance(belief_args, dict):
                belief_args = {str(k): v for k, v in belief_args.items()}
            else:
                belief_args = {}

            # 为缺少帖子或无 belief 的用户回退到初始化 belief
            try:
                df_belief = df_init_belief.copy()
                df_belief["user_id"] = df_belief["user_id"].astype(str)
                _fallback_no_post = 0
                _fallback_missing_belief = 0
                for uid in all_user:
                    uid_str = str(uid)
                    posts = belief_args.get(uid_str)
                    if not posts:
                        init_series = df_belief.loc[
                            df_belief["user_id"] == uid_str, "belief"
                        ]
                        if not init_series.empty:
                            belief_args[uid_str] = [{"belief": init_series.iloc[0]}]
                            _fallback_no_post += 1
                    else:
                        # 若存在帖子，但首条缺少 belief 字段，亦回退
                        if not isinstance(posts[0], dict) or "belief" not in posts[0]:
                            init_series = df_belief.loc[
                                df_belief["user_id"] == uid_str, "belief"
                            ]
                            if not init_series.empty:
                                belief_args[uid_str] = [{"belief": init_series.iloc[0]}]
                                _fallback_missing_belief += 1
                # 仅输出一次汇总信息
                if _fallback_no_post > 0 or _fallback_missing_belief > 0:
                    logging.info(
                        f"[belief] 回退汇总：当日无帖子 { _fallback_no_post } 人；帖子缺少 belief 字段 { _fallback_missing_belief } 人"
                    )
            except Exception:
                # 回退构建失败不致命，继续使用已有的 belief_args
                logging.warning(
                    "[belief] 构建 belief 回退映射时发生异常，已跳过回退构建"
                )
            if isinstance(belief_args, dict):
                # 统一 key 为字符串，避免 int/str 不一致导致查不到
                belief_args = {str(k): v for k, v in belief_args.items()}
        else:
            # 第一天：使用初始化的信念值数据
            belief_args = df_init_belief

        # ============================ 用户关系网络构建 ============================
        # 构建当前日期的用户关系网络图
        current_user_graph = build_graph_new(
            similarity_threshold=similarity_threshold,  # 相似度阈值
            time_decay_factor=time_decay_factor,  # 时间衰减因子
            db_path=user_db,  # 用户数据库路径
            start_date="2023-01-01",  # 图构建起始日期
            end_date=current_date.strftime("%Y-%m-%d"),  # 图构建结束日期
            save_name=f'{user_graph_save_name}_{current_date.strftime("%Y-%m-%d")}',  # 保存名称
            save=True,  # 保存图文件
        )
        print(
            f"用户关系图属性: {current_user_graph.number_of_nodes()} 个节点, {current_user_graph.number_of_edges()} 条边"
        )

        # 根据节点度数获取顶级用户列表
        top_user = get_top_n_users_by_degree(
            G=current_user_graph, top_n=int(node * top_n_user)
        )

        # ============================ 并发处理用户输入 ============================
        print(f"开始处理 {len(all_user)} 个用户，使用 {max_workers} 个工作线程...")

        # 初始化结果存储列表
        results = []  # 交易决策结果
        forum_args_list = []  # 论坛互动参数
        post_args_list = []  # 帖子发布参数

        # 使用线程池并发处理所有用户
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个用户创建并发任务
            futures = [
                executor.submit(
                    process_user_input,
                    user_id,
                    user_db,
                    forum_db,
                    df_stock,
                    current_date,
                    debug,
                    day_1st,
                    current_user_graph,
                    import_news,
                    df_strategy,
                    is_trading_day,
                    top_user,
                    log_dir,
                    prob_of_technical,
                    user_config_mapping,
                    activate_maapping,
                    belief_args,
                    user_config_mapping[user_id],  # 传入用户对应的配置
                )
                for user_id in all_user
            ]

            # 使用tqdm显示处理进度
            for future in tqdm(
                as_completed(futures),
                total=len(all_user),
                desc=f"处理输入 {current_date.strftime('%Y-%m-%d')}",
                unit="用户",
            ):
                try:
                    # 等待任务完成，设置超时限制
                    user_id, forum_args, decision_result, post_response_args = (
                        future.result(timeout=TIMEOUT_THRESHOLD)
                    )
                    # 收集各类处理结果
                    forum_args_list.append((user_id, forum_args))
                    results.append((user_id, decision_result))
                    post_args_list.append((user_id, post_response_args))

                except TimeoutError:
                    # 超时处理：使用备用配置重试
                    print(
                        f"[输入处理] 用户 {user_id} 超时: 处理超过 {TIMEOUT_THRESHOLD//60} 分钟。使用备用配置重试..."
                    )
                    fallback_config_path = "./config/api.yaml"
                    retry_future = executor.submit(
                        process_user_input,
                        user_id,
                        user_db,
                        forum_db,
                        df_stock,
                        current_date,
                        debug,
                        day_1st,
                        current_user_graph,
                        import_news,
                        df_strategy,
                        is_trading_day,
                        top_user,
                        log_dir,
                        prob_of_technical,
                        user_config_mapping,
                        activate_maapping,
                        belief_args,
                        fallback_config_path,
                    )
                    try:
                        # 重试任务的结果处理
                        user_id, forum_args, decision_result, post_response_args = (
                            retry_future.result(timeout=TIMEOUT_THRESHOLD)
                        )
                        forum_args_list.append((user_id, forum_args))
                        results.append((user_id, decision_result))
                        post_args_list.append((user_id, post_response_args))
                    except Exception as e:
                        print(f"[输入处理] 重试后仍失败，用户 {user_id}: {e}")

                except Exception as e:
                    # 其他异常处理
                    print(f"[输入处理] 处理用户 {user_id} 时出错: {e}")

        # ============================ 结果文件保存 ============================
        # 保存交易决策结果
        result_dir = os.path.join(f"{log_dir}/trading_records")
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(
            result_dir, f"{current_date.strftime('%Y-%m-%d')}.json"
        )
        with open(result_file, "w", encoding="utf-8") as f:
            result_dict = {user_id: result for user_id, result in results}
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        # 保存论坛反应记录
        reaction_result_dir = os.path.join(f"{log_dir}/reaction_records")
        os.makedirs(reaction_result_dir, exist_ok=True)
        reaction_result_file = os.path.join(
            reaction_result_dir, f"{current_date.strftime('%Y-%m-%d')}.json"
        )
        with open(reaction_result_file, "w", encoding="utf-8") as f:
            reaction_result_dict = {
                user_id: reaction_result for user_id, reaction_result in forum_args_list
            }
            json.dump(reaction_result_dict, f, indent=4, ensure_ascii=False)

        # 保存帖子发布记录
        post_result_dir = os.path.join(f"{log_dir}/post_records")
        os.makedirs(post_result_dir, exist_ok=True)
        post_result_file = os.path.join(
            post_result_dir, f"{current_date.strftime('%Y-%m-%d')}.json"
        )
        with open(post_result_file, "w", encoding="utf-8") as f:
            post_result_dict = {
                user_id: post_result for user_id, post_result in post_args_list
            }
            json.dump(post_result_dict, f, indent=4, ensure_ascii=False)

        # ============================ 论坛帖子处理 ============================
        # 统计成功创建的帖子数量
        successful_posts = 0
        if post_args_list:
            print(f"开始处理 {len(post_args_list)} 个用户的帖子发布...")
            for user_id, post_response_args in post_args_list:
                try:
                    # 未激活用户：仅提示一次 INFO，并跳过
                    if not activate_maapping.get(user_id, True):
                        logging.info(f"[inactive] 用户 {user_id} 未激活，跳过发帖")
                        continue

                    # 跳过无效的帖子响应（未激活用户或上游出错）
                    if not isinstance(post_response_args, dict):
                        continue

                    required_keys = ("post", "type", "belief")
                    if not all(
                        k in post_response_args and post_response_args[k] is not None
                        for k in required_keys
                    ):
                        continue

                    # 在论坛数据库中创建新帖子
                    create_post_db(
                        user_id=user_id,
                        content=post_response_args["post"],  # 帖子内容
                        type=post_response_args["type"],  # 帖子类型
                        belief=str(post_response_args["belief"]),  # 用户信念值
                        created_at=current_date,  # 创建时间
                        db_path=forum_db,  # 论坛数据库路径
                    )
                    successful_posts += 1
                except Exception as e:
                    print(f"[帖子处理] 用户 {user_id} 帖子创建失败: {e}")

            print(f"成功处理 {successful_posts}/{len(post_args_list)} 个用户的帖子发布")

        # ============================ 交易系统更新 ============================
        # 根据是否为交易日选择不同的更新策略
        if is_trading_day:
            # 交易日：运行交易匹配系统，处理所有交易请求
            test_matching_system(
                current_date=current_date.strftime("%Y-%m-%d"),
                base_path=log_dir,
                db_path=user_db,
                json_file_path=f"{log_dir}/trading_records/{current_date.strftime('%Y-%m-%d')}.json",
            )
        else:
            # 非交易日：仅更新用户资料表（不处理交易）
            update_profiles_table_holiday(
                current_date=current_date.strftime("%Y-%m-%d"), db_path=user_db
            )

        # ============================ 论坛互动处理 ============================
        if not day_1st:
            # 非第一天：处理用户在论坛中的互动行为
            successful_actions = 0
            print(f"开始处理 {len(forum_args_list)} 个用户的论坛互动...")

            if forum_args_list:
                for user_id, forum_args in forum_args_list:
                    try:
                        # 检查论坛参数是否有效
                        if isinstance(forum_args, str):
                            print(
                                f"[论坛互动] 用户 {user_id} 的论坛参数错误: {forum_args}"
                            )
                            continue

                        # 异步执行论坛互动操作（点赞、评论等）
                        asyncio.run(
                            execute_forum_actions(
                                forum_args=forum_args,
                                db_path=forum_db,
                                user_id=user_id,
                                created_at=current_date,
                            )
                        )
                        successful_actions += 1
                    except Exception as e:
                        print(f"[论坛互动] 用户 {user_id} 处理失败: {e}")

            print(
                f"成功处理 {successful_actions}/{len(forum_args_list)} 个用户的论坛互动"
            )

            # 更新论坛帖子的评分（基于互动数据）
            update_posts_score_by_date_range(
                db_path=forum_db, end_date=current_date.strftime("%Y-%m-%d")
            )

        # 日期递增，进入下一天的模拟
        current_date += timedelta(days=1)


def parse_args():
    """
    解析命令行参数，配置模拟系统的各种参数

    Returns:
        argparse.Namespace: 包含所有配置参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="初始化并运行股票交易模拟系统")

    # ============================ 时间范围配置 ============================
    parser.add_argument(
        "--start_date",
        type=str,
        default="2023-06-15",
        help="模拟开始日期 (格式: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2023-12-15",
        help="模拟结束日期 (格式: YYYY-MM-DD)",
    )

    # ============================ 数据库配置 ============================
    parser.add_argument(
        "--forum_db",
        type=str,
        default="logs_100_0128_claude/forum_100.db",
        help="论坛数据库文件路径",
    )
    parser.add_argument(
        "--user_db",
        type=str,
        default="logs_100_0128_claude/user_100.db",
        help="用户数据库文件路径",
    )

    # ============================ 运行配置 ============================
    parser.add_argument("--debug", type=bool, default=False, help="是否开启调试模式")
    parser.add_argument(
        "--max_workers", type=int, default=50, help="并发处理的最大线程数"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs_100_0128_claude", help="日志文件保存目录"
    )

    # ============================ 用户关系图配置 ============================
    parser.add_argument(
        "--user_graph_save_name",
        type=str,
        default="user_graph_logs_100_0128_claude",
        help="用户关系图保存文件名称",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.2,
        help="构建用户关系图的相似度阈值",
    )
    parser.add_argument(
        "--time_decay_factor",
        type=float,
        default=0.5,
        help="构建用户关系图的时间衰减因子",
    )
    parser.add_argument("--node", type=int, default=100, help="用户关系图中的节点数量")
    parser.add_argument(
        "--top_n_user", type=float, default=0.1, help="顶级用户所占比例"
    )

    # ============================ 交易行为配置 ============================
    parser.add_argument(
        "--prob_of_technical",
        type=float,
        default=0.5,
        help="技术面噪声交易者的激活概率",
    )
    parser.add_argument(
        "--activate_prob", type=float, default=1.0, help="用户激活参与模拟的概率"
    )

    # ============================ 数据文件配置 ============================
    parser.add_argument(
        "--belief_init_path",
        type=str,
        default="util/belief/belief_1000_0129.csv",
        help="用户初始信念值文件路径",
    )
    parser.add_argument(
        "--config_path", type=str, default="./config/api.yaml", help="API配置文件路径"
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    主程序入口：解析参数并启动模拟系统
    """

    # ============================ 参数解析与验证 ============================
    args = parse_args()

    # 确保日志目录存在
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        print(f"创建日志目录: {args.log_dir}")

    # 输出所有配置参数
    print("\n=== 模拟配置参数 ===")
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))

    # ============================ 检查点逻辑判断 ============================
    # 根据开始日期判断是否从检查点开始
    if args.start_date == "2023-06-15":
        checkpoint = False  # 从头开始，需要初始化所有数据
        print("模式: 从头开始模拟")
    else:
        checkpoint = True  # 从检查点继续，使用已有数据
        print(f"模式: 从检查点 {args.start_date} 继续模拟")

    # ============================ 数据库初始化 ============================
    if not checkpoint:
        print("初始化论坛数据库...")
        init_db_forum(db_path=args.forum_db)
        print("论坛数据库初始化完成")

    # ============================ 启动模拟系统 ============================
    print("\n=== 开始运行交易模拟系统 ===")
    init_simulation(
        start_date=pd.Timestamp(args.start_date),
        end_date=pd.Timestamp(args.end_date),
        forum_db=args.forum_db,
        user_db=args.user_db,
        debug=args.debug,
        max_workers=args.max_workers,
        user_graph_save_name=args.user_graph_save_name,
        checkpoint=checkpoint,
        similarity_threshold=args.similarity_threshold,
        time_decay_factor=args.time_decay_factor,
        node=args.node,
        log_dir=args.log_dir,
        prob_of_technical=args.prob_of_technical,
        belief_init_path=args.belief_init_path,
        config_path=args.config_path,
        activate_prob=args.activate_prob,
    )
    print("\n=== 模拟系统运行完成 ===")

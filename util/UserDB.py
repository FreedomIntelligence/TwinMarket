import numpy as np
import json
import datetime
import os
import sqlite3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
from typing import Optional
from collections import defaultdict
from datetime import datetime
from . import IndustryDict

DB_PATH = 'data/UserDB/sys_100.db'


def get_top_industry_and_category(user_id, db_path=DB_PATH):
    """Get the most frequently traded industry and its categories for a user."""

    def find_category_ch(industry):
        for category, industries in IndustryDict.ch.items():
            if industry in industries:
                return category
        return "未知类别"

    def find_category_eng(industry):
        for category, industries in IndustryDict.eng.items():
            if industry in industries:
                return category
        return "Unknown"

    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询用户的交易记录及对应的行业
    query = '''
    SELECT industry, COUNT(industry) AS count
    FROM TradingDetails
    WHERE user_id = ?
    GROUP BY industry
    ORDER BY count DESC
    LIMIT 1
    '''
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    # 如果没有交易记录，返回 None
    if result is None:
        return None, None, None

    # 获取行业及其类别
    top_industry, _ = result
    category_ch = find_category_ch(top_industry)
    category_eng = find_category_eng(top_industry)
    conn.close()
    return top_industry, category_ch, category_eng


def get_user_profile(user_id: str, db_path: str = DB_PATH, created_at: str = None) -> dict:
    """
    获取用户的详细信息，并返回一个包含用户信息的字典。

    Args:
        user_id (str): 用户的唯一标识符。
        db_path (str): 数据库文件路径，默认为 DB_PATH。

    Returns:
        dict: 包含用户详细信息的字典。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询用户的基本信息和投资数据，包括 created_at
    query = '''
        SELECT gender, location, user_type,
               bh_disposition_effect_category, bh_lottery_preference_category,
               bh_total_return_category, bh_annual_turnover_category,
               bh_underdiversification_category, trade_count_category,
               sys_prompt, prompt, self_description,
               trad_pro, fol_ind, ini_cash, initial_positions,
               current_cash, cur_positions, total_value,
               total_return, return_rate, stock_returns, yest_returns,
               created_at
        FROM Profiles
        WHERE user_id = ? AND created_at = ?'''
    cursor.execute(query, (user_id, created_at))

    result = cursor.fetchone()

    # 如果没有找到用户，返回空字典
    if result is None:
        conn.close()
        return {}

    # 将查询结果映射到字典中
    user_profile = {
        'gender': result[0],
        'location': result[1],
        'user_type': result[2],
        'bh_disposition_effect_category': result[3],
        'bh_lottery_preference_category': result[4],
        'bh_total_return_category': result[5],
        'bh_annual_turnover_category': result[6],
        'bh_underdiversification_category': result[7],
        'trade_count_category': result[8],
        'sys_prompt': result[9],
        'prompt': result[10],
        'self_description': result[11],
        'trad_pro': result[12],
        'fol_ind': json.loads(result[13]) if result[13] else [],  # 处理 fol_ind 字段
        'ini_cash': result[14],
        'initial_positions': json.loads(result[15]) if result[15] else None,  # 处理 initial_positions 字段
        'current_cash': result[16],
        'cur_positions': json.loads(result[17]) if result[17] else None,  # 处理 cur_positions 字段
        'total_value': result[18],
        'total_return': result[19],
        'return_rate': result[20],
        'stock_returns': json.loads(result[21]) if result[21] else None,  # 处理 stock_returns 字段
        'yest_returns': json.loads(result[22]) if result[22] else None,  # 处理 yest_returns 字段
        'created_at': result[23],  # 新增 created_at 字段
        'user_id': user_id
    }

    # 处理 fol_ind 字段，确保解析为列表并还原中文字符
    fol_ind = result[13]
    if fol_ind:
        try:
            # 解析 JSON 并确保中文字符正确还原
            user_profile['fol_ind'] = json.loads(fol_ind)
        except json.JSONDecodeError:
            # 如果解析失败，尝试直接处理为列表
            if isinstance(fol_ind, str) and fol_ind.startswith('[') and fol_ind.endswith(']'):
                user_profile['fol_ind'] = [item.strip().strip('"') for item in fol_ind[1:-1].split(',')]
            else:
                user_profile['fol_ind'] = [fol_ind]
            print(f"警告: fol_ind 字段不是有效的 JSON 格式，已尝试转换为列表。")
    else:
        user_profile['fol_ind'] = []

    # 处理 JSON 格式的字段
    json_fields = {
        'initial_positions': result[15],
        'cur_positions': result[17],
        'stock_returns': result[21],
        'yest_returns': result[22]
    }

    for field, value in json_fields.items():
        try:
            # 尝试将字符串解析为 JSON，并确保中文字符正确还原
            user_profile[field] = json.loads(value) if value else None
        except json.JSONDecodeError:
            # 如果解析失败，保留原始值
            user_profile[field] = value
            print(f"警告: 字段 {field} 不是有效的 JSON 格式，保留原始值。")

    conn.close()
    return user_profile


def get_user_trading_records(user_id: str, start_date: str = None, end_date: str = None, db_path=DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get user's trading records and daily trade counts."""
    conn = sqlite3.connect(db_path)
    try:
        if start_date is None and end_date is None:
            query = """
            SELECT date_time, stock_id, stock_name, price, trading_direction, volume, industry
            FROM TradingDetails 
            WHERE user_id = ? 
            ORDER BY date_time
            """
            df = pd.read_sql_query(query, conn, params=(user_id,))
        else:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime(1970, 1, 1)
                end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            except ValueError:
                raise ValueError("Dates must be in YYYY-MM-DD format")

            query = """
            SELECT date_time, stock_id, stock_name, price, trading_direction, volume, industry
            FROM TradingDetails 
            WHERE user_id = ? 
            AND datetime(date_time) BETWEEN datetime(?) AND datetime(?)
            ORDER BY date_time
            """
            df = pd.read_sql_query(query, conn, params=(user_id, start_dt.isoformat(), end_dt.isoformat()))

        df['date_time'] = pd.to_datetime(df['date_time'])
        daily_counts = df.groupby(df['date_time'].dt.strftime('%Y%m%d')).size().reset_index()
        daily_counts.columns = ['time', 'trades_count']
        daily_counts = daily_counts.sort_values('time').reset_index(drop=True)

        return df, daily_counts
    finally:
        conn.close()


def get_all_user_ids(db_path: str = DB_PATH, timestamp: Optional[pd.Timestamp] = None) -> list:
    """
    从数据库中获取所有用户的 user_id。

    Args:
        db_path (str): 数据库文件路径，默认为 DB_PATH。
        timestamp (Optional[pd.Timestamp]): 时间戳，用于过滤用户。如果为 None，则返回所有用户。

    Returns:
        list: 包含所有 user_id 的列表。
    """
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 构建查询语句
        if timestamp is not None:
            # 如果提供了时间戳，过滤 created_at 小于等于该时间戳的记录
            query = "SELECT DISTINCT user_id FROM Profiles WHERE created_at <= ?"
            cursor.execute(query, (timestamp.strftime('%Y-%m-%d %H:%M:%S'),))
        else:
            # 如果没有提供时间戳，返回所有用户
            query = "SELECT DISTINCT user_id FROM Profiles"
            cursor.execute(query)

        # 提取 user_id 并返回列表
        results = cursor.fetchall()
        user_ids = [row[0] for row in results]
        conn.close()
        return user_ids

    except Exception as e:
        print(f"Error fetching user IDs: {e}")
        return []


def save_graph(G: nx.Graph,
               filename: str,
               output_dir: str = 'data/UserDB',
               format: str = 'pickle') -> bool:
    """
    Save NetworkX graph to file in GraphML and/or Pickle format

    Args:
        G (nx.Graph): NetworkX graph to save
        filename (str): Base filename without extension
        output_dir (str): Directory to save files
        format (str): 'graphml', 'pickle' or 'both'

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        if format in ['graphml', 'both']:
            graphml_path = os.path.join(output_dir, f"{filename}.graphml")
            nx.write_graphml(G, graphml_path)

        if format in ['pickle', 'both']:
            pickle_path = os.path.join(output_dir, f"{filename}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(G, f)

        return True

    except Exception as e:
        print(f"Error saving graph: {e}")
        return False


def load_graph(filename: str,
               input_dir: str = 'data/UserDB',
               format: str = 'pickle') -> Optional[nx.Graph]:
    """
    Load NetworkX graph from file

    Args:
        filename (str): Base filename without extension
        input_dir (str): Directory containing graph files
        format (str): 'graphml' or 'pickle'

    Returns:
        Optional[nx.Graph]: Loaded graph or None if failed
    """
    try:
        if format == 'graphml':
            path = os.path.join(input_dir, f"{filename}.graphml")
            return nx.read_graphml(path)
        else:
            path = os.path.join(input_dir, f"{filename}.pkl")
            with open(path, 'rb') as f:
                return pickle.load(f)

    except Exception as e:
        print(f"Error loading graph: {e}")
        return None


# def build_graph(
#     sparsity_factor: float = 0.15,
#     db_path: str = DB_PATH,
#     save: bool = True,
#     save_name: str = 'user_graph',
#     start_date: str = '2023-01-01',
#     end_date: str = '2023-12-31'
# ) -> nx.Graph:
#     """
#     基于用户交易相似性构建稀疏图。
#     自动获取所有用户的交易记录。

#     参数:
#         sparsity_factor: 保留边的比例 (0-1)
#         db_path: 数据库路径
#         save: 是否保存图
#         save_name: 保存的文件名
#         start_date: 交易记录的开始日期
#         end_date: 交易记录的结束日期

#     返回:
#         带有用户相似性和行业属性的 NetworkX 图

#     异常:
#         ValueError: 如果无法获取交易记录或数据无效
#     """
#     try:
#         # 获取所有用户ID
#         conn = sqlite3.connect(db_path)
#         user_ids = pd.read_sql_query("SELECT DISTINCT user_id FROM Profiles", conn)['user_id'].tolist()
#         conn.close()

#         # 获取所有用户的交易记录
#         trading_records_df_list = []
#         for user_id in user_ids:
#             user_trading_records_df, _ = get_user_trading_records(
#                 user_id=user_id,
#                 db_path=db_path,
#                 start_date=start_date,
#                 end_date=end_date
#             )
#             if user_trading_records_df is not None:
#                 user_trading_records_df['user_id'] = user_id
#                 trading_records_df_list.append(user_trading_records_df)

#         # 合并所有用户的交易记录
#         trading_records_df = pd.concat(trading_records_df_list, ignore_index=True)

#         # 获取每个用户的股票组合
#         user_portfolios = {}
#         for user_id in user_ids:
#             user_trading_records = trading_records_df[trading_records_df['user_id'] == user_id]
#             if not user_trading_records.empty:
#                 user_portfolios[user_id] = set(user_trading_records['stock_id'])
#             else:
#                 user_portfolios[user_id] = set()  # 如果用户没有交易记录，分配空集合

#         # 计算用户之间的相似性得分
#         edge_scores = []
#         for user1, stocks1 in user_portfolios.items():
#             for user2, stocks2 in user_portfolios.items():
#                 if user1 >= user2:
#                     continue

#                 # 计算 Jaccard 相似性
#                 intersection = len(stocks1 & stocks2)
#                 union = len(stocks1 | stocks2)
#                 if union == 0:
#                     continue

#                 similarity = intersection / union
#                 if similarity > 0:
#                     edge_scores.append((user1, user2, similarity))

#         # 按得分排序
#         edge_scores.sort(key=lambda x: x[2], reverse=True)

#         # 构建图
#         G = nx.Graph()
#         G.add_nodes_from(user_ids)  # 将所有用户添加为节点

#         # 确保每个节点至少有一条边
#         connected_nodes = set()
#         for user1, user2, score in edge_scores:
#             if user1 not in connected_nodes or user2 not in connected_nodes:
#                 G.add_edge(user1, user2, weight=score)
#                 connected_nodes.add(user1)
#                 connected_nodes.add(user2)

#         # 处理孤立节点
#         isolated_nodes = set(user_ids) - connected_nodes
#         for user in isolated_nodes:
#             # 找到与当前用户相似性最高的用户
#             best_match = None
#             best_score = 0
#             for other_user in connected_nodes:
#                 if other_user in user_portfolios:
#                     stocks1 = user_portfolios[user]
#                     stocks2 = user_portfolios[other_user]
#                     intersection = len(stocks1 & stocks2)
#                     union = len(stocks1 | stocks2)
#                     if union == 0:
#                         continue
#                     similarity = intersection / union
#                     if similarity > best_score:
#                         best_score = similarity
#                         best_match = other_user

#             # 添加边
#             if best_match is not None:
#                 G.add_edge(user, best_match, weight=best_score)
#                 connected_nodes.add(user)

#         # 根据稀疏因子保留剩余的边
#         num_edges_to_keep = int(len(edge_scores) * sparsity_factor)
#         for i in range(len(edge_scores)):
#             if len(G.edges) >= num_edges_to_keep:
#                 break
#             user1, user2, score = edge_scores[i]
#             if not G.has_edge(user1, user2):
#                 G.add_edge(user1, user2, weight=score)

#         # 为每个节点添加行业和类别属性
#         for user_id in G.nodes():
#             industry, category_ch, category_eng = get_top_industry_and_category(user_id=user_id, db_path=db_path)
#             if industry is None or category_ch is None or category_eng is None:
#                 industry = industry if industry else '未知'
#                 category_ch = category_ch if category_ch else '未知'
#                 category_eng = category_eng if category_eng else 'Unknown'
#             G.nodes[user_id]['industry'] = industry
#             G.nodes[user_id]['category_ch'] = category_ch
#             G.nodes[user_id]['category_eng'] = category_eng

#         # 保存图
#         if save:
#             save_graph(G=G, filename=save_name, output_dir='data/UserDB', format='pickle')

#         return G

#     except Exception as e:
#         raise ValueError(f"Failed to build user similarity graph: {str(e)}")


def build_graph(
    db_path: str = DB_PATH,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    similarity_threshold: float = 0.1,  # 相似性阈值
    save: bool = True,  # 是否保存图
    save_name: str = 'user_graph',  # 保存的文件名
    output_dir: str = 'data/UserDB/graph'  # 保存的目录
) -> nx.Graph:
    """
    基于用户交易相似性构建图，并保留节点的属性。

    参数:
        db_path: 数据库路径
        start_date: 交易记录的开始日期
        end_date: 交易记录的结束日期
        similarity_threshold: 相似性阈值，只有超过该值的相似性才会被添加为边
        save: 是否保存图
        save_name: 保存的文件名
        output_dir: 保存的目录

    返回:
        带有用户相似性和节点属性的 NetworkX 图

    异常:
        ValueError: 如果无法获取交易记录或数据无效
    """
    try:
        # 获取所有用户ID
        conn = sqlite3.connect(db_path)
        user_ids = pd.read_sql_query("SELECT DISTINCT user_id FROM Profiles", conn)['user_id'].tolist()
        conn.close()

        # 获取所有用户的交易记录
        trading_records_df_list = []
        for user_id in user_ids:
            user_trading_records_df, _ = get_user_trading_records(
                user_id=user_id,
                db_path=db_path,
                start_date=start_date,
                end_date=end_date
            )
            if user_trading_records_df is not None:
                user_trading_records_df['user_id'] = user_id
                trading_records_df_list.append(user_trading_records_df)

        # 合并所有用户的交易记录
        trading_records_df = pd.concat(trading_records_df_list, ignore_index=True)

        # 获取每个用户的股票组合
        user_portfolios = {}
        for user_id in user_ids:
            user_trading_records = trading_records_df[trading_records_df['user_id'] == user_id]
            if not user_trading_records.empty:
                user_portfolios[user_id] = set(user_trading_records['stock_id'])
            else:
                user_portfolios[user_id] = set()  # 如果用户没有交易记录，分配空集合

        # 计算用户之间的相似性得分
        edge_scores = []
        for user1, stocks1 in user_portfolios.items():
            for user2, stocks2 in user_portfolios.items():
                if user1 >= user2:
                    continue

                # 计算 Jaccard 相似性
                intersection = len(stocks1 & stocks2)
                union = len(stocks1 | stocks2)
                if union == 0:
                    continue

                similarity = intersection / union
                if similarity > similarity_threshold:  # 只保留超过阈值的相似性
                    edge_scores.append((user1, user2, similarity))

        # 构建图
        G = nx.Graph()
        G.add_nodes_from(user_ids)  # 将所有用户添加为节点

        # 添加边
        for user1, user2, score in edge_scores:
            G.add_edge(user1, user2, weight=score)

        # 为每个节点添加行业和类别属性
        for user_id in G.nodes():
            industry, category_ch, category_eng = get_top_industry_and_category(user_id=user_id, db_path=db_path)
            if industry is None or category_ch is None or category_eng is None:
                industry = industry if industry else '未知'
                category_ch = category_ch if category_ch else '未知'
                category_eng = category_eng if category_eng else 'Unknown'
            G.nodes[user_id]['industry'] = industry
            G.nodes[user_id]['category_ch'] = category_ch
            G.nodes[user_id]['category_eng'] = category_eng

        # 保存图
        if save:
            save_graph(G=G, filename=save_name, output_dir=output_dir, format='pickle')

        return G

    except Exception as e:
        raise ValueError(f"Failed to build user similarity graph: {str(e)}")
    

def get_top_n_users_by_degree(G: nx.Graph, top_n: int) -> list:
    """
    返回图中度数最高的 top_n 个用户 ID。

    参数:
        G: NetworkX 图
        top_n: 需要返回的最高度数用户的数量

    返回:
        包含 top_n 个最高度数用户 ID 的列表
    """
    # 获取所有节点的度数
    degrees = dict(G.degree())

    # 按度数从高到低排序，并提取对应的 user_id
    sorted_users = sorted(
        G.nodes(data=True),  # 获取节点及其属性
        key=lambda x: degrees[x[0]],  # 按度数排序
        reverse=True  # 降序排列
    )

    # 提取前 top_n 个用户的 user_id
    top_n_users = [user[0] for user in sorted_users[:top_n]]

    return top_n_users


def build_graph_new(
    db_path: str = DB_PATH,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    similarity_threshold: float = 0.1,  # 相似性阈值
    time_decay_factor: float = 0.1,  # 时间衰减因子
    save: bool = True,  # 是否保存图
    save_name: str = 'user_graph',  # 保存的文件名
    output_dir: str = 'data/UserDB/graph'  # 保存的目录
) -> nx.Graph:
    """
    基于用户购买股票的行业相似性构建图，并保留节点的属性。
    考虑时间衰减的影响。

    参数:
        db_path: 数据库路径
        start_date: 交易记录的开始日期
        end_date: 交易记录的结束日期
        similarity_threshold: 相似性阈值，只有超过该值的相似性才会被添加为边
        time_decay_factor: 时间衰减因子，控制时间对相似性的影响
        save: 是否保存图
        save_name: 保存的文件名
        output_dir: 保存的目录

    返回:
        带有用户相似性和节点属性的 NetworkX 图

    异常:
        ValueError: 如果无法获取交易记录或数据无效
    """
    try:
        # 获取所有用户ID
        conn = sqlite3.connect(db_path)
        user_ids = pd.read_sql_query("SELECT DISTINCT user_id FROM Profiles", conn)['user_id'].tolist()
        conn.close()

        # 获取所有用户的交易记录
        trading_records_df_list = []
        for user_id in user_ids:
            user_trading_records_df, _ = get_user_trading_records(
                user_id=user_id,
                db_path=db_path,
                start_date=start_date,
                end_date=end_date
            )
            if user_trading_records_df is not None:
                user_trading_records_df['user_id'] = user_id
                trading_records_df_list.append(user_trading_records_df)

        # 合并所有用户的交易记录
        trading_records_df = pd.concat(trading_records_df_list, ignore_index=True)

        # 将日期列转换为 datetime 类型
        trading_records_df['date_time'] = pd.to_datetime(trading_records_df['date_time'])

        # 获取每个用户购买的行业及其时间权重
        user_industries = defaultdict(lambda: defaultdict(float))  # {user_id: {industry: weighted_count}}
        current_time = datetime.now()  # 当前时间，用于计算时间衰减

        for _, row in trading_records_df.iterrows():
            user_id = row['user_id']
            industry = row['industry']
            trade_time = row['date_time']

            if industry:  # 确保行业信息不为空
                # 计算时间衰减权重
                time_diff = (current_time - trade_time).days  # 时间差（天数）
                time_weight = np.exp(-time_decay_factor * time_diff)  # 指数衰减

                # 更新用户行业的加权计数
                user_industries[user_id][industry] += time_weight

        # 计算用户之间的行业相似性得分（考虑时间衰减）
        edge_scores = []
        for user1, industries1 in user_industries.items():
            for user2, industries2 in user_industries.items():
                if user1 >= user2:
                    continue

                # 计算加权 Jaccard 相似性
                intersection = set(industries1.keys()) & set(industries2.keys())
                union = set(industries1.keys()) | set(industries2.keys())

                if not union:
                    continue

                # 计算加权交集和并集
                weighted_intersection = sum(min(industries1[industry], industries2[industry]) for industry in intersection)
                weighted_union = sum(max(industries1.get(industry, 0), industries2.get(industry, 0)) for industry in union)

                similarity = weighted_intersection / weighted_union if weighted_union > 0 else 0

                if similarity > similarity_threshold:  # 只保留超过阈值的相似性
                    edge_scores.append((user1, user2, similarity))

        # 构建图
        G = nx.Graph()
        G.add_nodes_from(user_ids)  # 将所有用户添加为节点

        # 添加边
        for user1, user2, score in edge_scores:
            G.add_edge(user1, user2, weight=score)

        # 检查是否有孤立节点，如果有则为其添加边，权重为最小值
        isolated_nodes = list(nx.isolates(G))  # 获取所有孤立节点
        if isolated_nodes:
            min_weight = min(score for _, _, score in edge_scores) if edge_scores else similarity_threshold
            for i in range(len(isolated_nodes) - 1):
                G.add_edge(isolated_nodes[i], isolated_nodes[i + 1], weight=min_weight)

        # 为每个节点添加行业和类别属性
        for user_id in G.nodes():
            industry, category_ch, category_eng = get_top_industry_and_category(user_id=user_id, db_path=db_path)
            if industry is None or category_ch is None or category_eng is None:
                industry = industry if industry else '未知'
                category_ch = category_ch if category_ch else '未知'
                category_eng = category_eng if category_eng else 'Unknown'
            G.nodes[user_id]['industry'] = industry
            G.nodes[user_id]['category_ch'] = category_ch
            G.nodes[user_id]['category_eng'] = category_eng

        # 保存图
        if save:
            save_graph(G=G, filename=save_name, output_dir=output_dir, format='pickle')

        return G

    except Exception as e:
        raise ValueError(f"Failed to build user similarity graph: {str(e)}")


def update_graph(
    G: nx.Graph,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    db_path: str = DB_PATH,
    sparsity_factor: float = 0.15
) -> nx.Graph:
    """
    基于新的日期范围更新图中的节点和边，允许新的边被添加。

    参数:
        G: 现有的 NetworkX 图。
        start_date: 新的开始日期（格式为 'YYYY-MM-DD'）。
        end_date: 新的结束日期（格式为 'YYYY-MM-DD'）。
        db_path: 数据库路径。
        sparsity_factor: 新增边的稀疏因子（0-1），控制新增边的比例。

    返回:
        更新后的 NetworkX 图。

    异常:
        ValueError: 如果无法获取交易记录或数据无效。
    """
    try:
        # 获取现有图中的所有用户ID
        existing_user_ids = set(G.nodes())

        # 获取所有用户ID（包括新用户）
        conn = sqlite3.connect(db_path)
        all_user_ids = pd.read_sql_query("SELECT DISTINCT user_id FROM Profiles", conn)['user_id'].tolist()
        conn.close()

        # 获取所有用户的交易记录
        trading_records_df_list = []
        for user_id in all_user_ids:
            user_trading_records_df, _ = get_user_trading_records(
                user_id=user_id,
                db_path=db_path,
                start_date=start_date,
                end_date=end_date
            )
            if user_trading_records_df is not None:
                user_trading_records_df['user_id'] = user_id
                trading_records_df_list.append(user_trading_records_df)

        # 合并所有用户的交易记录
        trading_records_df = pd.concat(trading_records_df_list, ignore_index=True)

        # 获取每个用户的股票组合
        user_portfolios = trading_records_df.groupby('user_id')['stock_id'].apply(set).to_dict()

        # 更新现有图的边权重
        for user1, user2 in G.edges():
            if user1 in user_portfolios and user2 in user_portfolios:
                stocks1 = user_portfolios[user1]
                stocks2 = user_portfolios[user2]

                # 计算 Jaccard 相似性
                intersection = len(stocks1 & stocks2)
                union = len(stocks1 | stocks2)
                if union == 0:
                    similarity = 0
                else:
                    similarity = intersection / union

                # 更新边权重
                G[user1][user2]['weight'] = similarity

        # 添加新用户（如果有）
        new_user_ids = set(all_user_ids) - existing_user_ids
        for new_user_id in new_user_ids:
            if new_user_id in user_portfolios:
                stocks_new = user_portfolios[new_user_id]

                # 计算新用户与现有用户的相似性
                for existing_user_id in existing_user_ids:
                    if existing_user_id in user_portfolios:
                        stocks_existing = user_portfolios[existing_user_id]

                        # 计算 Jaccard 相似性
                        intersection = len(stocks_new & stocks_existing)
                        union = len(stocks_new | stocks_existing)
                        if union == 0:
                            similarity = 0
                        else:
                            similarity = intersection / union

                        # 添加新边
                        if similarity > 0:
                            G.add_edge(new_user_id, existing_user_id, weight=similarity)

        # 计算新增边的候选列表
        new_edge_candidates = []
        for user1 in existing_user_ids:
            for user2 in existing_user_ids:
                if user1 >= user2:
                    continue

                if user1 in user_portfolios and user2 in user_portfolios:
                    stocks1 = user_portfolios[user1]
                    stocks2 = user_portfolios[user2]

                    # 计算 Jaccard 相似性
                    intersection = len(stocks1 & stocks2)
                    union = len(stocks1 | stocks2)
                    if union == 0:
                        similarity = 0
                    else:
                        similarity = intersection / union

                    # 如果边不存在且相似性大于 0，则添加到候选列表
                    if not G.has_edge(user1, user2) and similarity > 0:
                        new_edge_candidates.append((user1, user2, similarity))

        # 按相似性得分排序
        new_edge_candidates.sort(key=lambda x: x[2], reverse=True)

        # 根据稀疏因子保留新增边
        num_new_edges_to_keep = int(len(new_edge_candidates) * sparsity_factor)
        for i in range(len(new_edge_candidates)):
            if i >= num_new_edges_to_keep:
                break
            user1, user2, similarity = new_edge_candidates[i]
            G.add_edge(user1, user2, weight=similarity)

        # 为每个节点更新行业和类别属性
        for user_id in G.nodes():
            industry, category_ch, category_eng = get_top_industry_and_category(user_id=user_id, db_path=db_path)
            if industry is None or category_ch is None or category_eng is None:
                industry = industry if industry else '未知'
                category_ch = category_ch if category_ch else '未知'
                category_eng = category_eng if category_eng else 'Unknown'
            G.nodes[user_id]['industry'] = industry
            G.nodes[user_id]['category_ch'] = category_ch
            G.nodes[user_id]['category_eng'] = category_eng

        return G

    except Exception as e:
        raise ValueError(f"Failed to update graph: {str(e)}")


def visualize_graph(G, radius=5, width=0.1):
    # Industry-specific color mapping
    industry_colors = {
        'Manufacturing': '#FF6B6B',
        'Energy and Resources': '#4ECDC4',
        'Financial Services': '#45B7D1',
        'Infrastructure and Engineering': '#96CEB4',
        'Consumer Goods': '#FFEEAD',
        'Technology and Communication': '#FFB347',
        'Transportation and Logistics': '#A47786',
        'Real Estate': '#87CEEB',
        'Tourism and Services': '#98FB98',
        'Chemical and Pharmaceuticals': '#DDA0DD'
    }

    # Get node colors
    node_colors = [industry_colors.get(G.nodes[node].get('category_eng', 'other'), '#D3D3D3')
                   for node in G.nodes()]

    # Group nodes by industry
    industry_groups = {}
    for node in G.nodes():
        industry = G.nodes[node].get('category_eng', 'other')
        if industry not in industry_groups:
            industry_groups[industry] = []
        industry_groups[industry].append(node)

    # Create a custom layout to group nodes by industry
    pos = {}
    num_industries = len(industry_groups)
    angle_step = 2 * np.pi / num_industries  # Divide the circle into equal parts for each industry
    radius = radius  # Radius of the main circle

    for i, (industry, nodes) in enumerate(industry_groups.items()):
        # Calculate the center of the sub-circle for this industry
        center_x = radius * np.cos(i * angle_step)
        center_y = radius * np.sin(i * angle_step)

        # Create a subgraph for each industry
        subgraph = G.subgraph(nodes)
        # Use a circular layout for each industry group
        subgraph_pos = nx.circular_layout(subgraph, scale=0.3, center=(center_x, center_y))
        pos.update(subgraph_pos)

    # Create visualization
    plt.figure(figsize=(15, 15))

    # Draw network
    nx.draw(G, pos,
            node_size=50,
            node_color=node_colors,
            with_labels=False,
            width=0.1,
            alpha=0.8)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=industry, markerfacecolor=color, markersize=10)
                       for industry, color in industry_colors.items()]
    plt.legend(handles=legend_elements, loc='center left',
               bbox_to_anchor=(1, 0.5), title='Industries')

    plt.title('Industry Network Graph (Grouped by Industry)')
    plt.tight_layout()
    plt.show()

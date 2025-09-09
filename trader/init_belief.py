# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent.futures
import time
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agent2 import BaseAgent 



SYSTEM_PROMPT='当前时间点下，我认为未来1个月市场将呈现震荡调整趋势。从历史规律来看，市场在经历一段时间的上涨后，往往会出现回调或盘整，因此短期内需警惕可能的下跌风险。同时，市场也可能在关键支撑位附近企稳，形成新的上涨动力。从市场估值来看，我认为当前市场整体估值处于合理区间，但部分热门板块可能存在高估风险，需谨慎对待。对于宏观经济走势，我持中性态度，虽然经济复苏仍在持续，但通胀压力和货币政策的不确定性可能对市场造成一定压力。市场情绪方面，我认为当前市场情绪中性偏谨慎，投资者在乐观与悲观之间摇摆，情绪波动较大，需警惕“追涨杀跌”的心理陷阱。结合我的历史交易表现和投资风格，我认为自己的投资水平处于中等水平，能够通过技术分析和基本面研究捕捉部分机会，但情绪驱动的决策模式和较高的处置效应（如“过早卖出盈利股票，长期持有亏损股票”）限制了整体收益的提升。记住，市场总是周期波动的，涨多了会跌，跌多了会涨，保持冷静和耐心是关键。'

def read_from_db(db_path, table_name):
    """从SQLite数据库读取指定日期范围的数据
    
    Args:
        db_path (str): 数据库文件路径
        table_name (str): 表名
        
    """
    try:
        conn = sqlite3.connect(db_path)
        current_time='2023-06-14 00:00:00'
        query = f"SELECT * FROM {table_name} WHERE created_at ='2023-06-14 00:00:00' "
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            print(f"警告：没有找到任何数据")
        else:
            print(f"成功读取数据，共 {len(df)} 条记录")
            
        return df
    except Exception as e:
        print(f"读取数据库时发生错误: {e}")
        raise

def get_init_prompt(row, attitude):
    """
    根据用户信息和态度生成初始 prompt。

    Args:
        row (pd.Series): 用户信息的一行数据。
        attitude (str): 用户的态度（如 "稳健型"、"激进型" 等）。

    Returns:
        str: 生成的 prompt。
    """
    # 解析用户信息
    strategy = row['strategy']  # 投资风格
    disposition_effect = row['bh_disposition_effect_category']  # 处置效应
    lottery_preference = row['bh_lottery_preference_category']  # 彩票偏好
    diversification = row['bh_underdiversification_category']  # 分散投资
    total_return = row['total_return']  # 总投资回报
    return_rate = row['return_rate']  # 回报率
    stock_returns = row['stock_returns']  # 持仓股票表现
    fol_ind = row['fol_ind']  # 关注行业
    self_description = row['self_description']  # 自我描述

    # 生成 prompt
    prompt = f"""
    你是一位**{attitude}的投资者**，以下是你的投资特征和行为模式：

    1. **投资风格**：你是一位{strategy}投资者。
    2. **心理特征**：
       - 处置效应：{disposition_effect}（高处置效应意味着你倾向于过早卖出盈利股票，而长期持有亏损股票）。
       - 彩票偏好：{lottery_preference}（低彩票偏好意味着你对高风险、高回报的“彩票型”股票兴趣较低）。
       - 分散投资：{diversification}（低分散投资意味着你倾向于集中投资于少数行业或个股）。
    3. **投资表现**：
       - 总投资回报：{total_return}
       - 回报率：{return_rate}%。
    4. **自我描述(最重要）**：{self_description}
    5. **市场的一般规律（仅供参考）**：
       - 周期性波动：市场总是呈现周期性波动，涨多了会跌，跌多了会涨。
       - 情绪驱动：市场情绪往往在极度乐观和极度悲观之间摇摆。
       - 均值回归：热门板块或个股在经历大幅上涨后，往往会出现回调。
       - 风险与收益：高收益通常伴随高风险。
       - 长期趋势：短期市场波动难以预测，但长期趋势往往与经济基本面一致。

    请根据以上信息，以第一人称的方式，用自然语言描述你对市场的看法和自身的投资评价。请直接输出一段话，不需要任何额外的结构或标题。你的回答应当包含以下内容：
    - 你对未来1个月市场大方向的看法。
    - 你对当前市场估值的看法。
    - 你对未来宏观经济走势的看法。
    - 你对当前市场情绪的看法。
    - 你结合历史交易表现和投资风格，对自我投资水平的评价。

    请尽量让回答自然流畅，避免机械化的模板化表达，直接输出文本格式即可。
    """
    return prompt

def retry_belief_conversion(agent, row, attitude, max_retries=3, delay=1):
    """带重试机制的市场指标分析函数"""
    for attempt in range(max_retries):
        try:
            prompt= get_init_prompt(row,attitude)
            response = agent.get_response(user_input=prompt).get("response")
            # 解析JSON响应
            print(response)
            return str(response)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"分析文本失败 '{row['user_id']}': {e}")
                response = SYSTEM_PROMPT
                return response
            time.sleep(delay)
    return SYSTEM_PROMPT
     

def process_chunk(chunk, agent):
    """处理数据块的函数"""
    chunk_copy = chunk.copy()
    # 为每个指标创建新列
    
    chunk_copy['belief'] = ''
    chunk_copy['attitude'] = ''
    # 处理每行数据
    for idx, row in chunk_copy.iterrows():
        attitude = random.choices(
            ['乐观的', '对市场态度中性的', '悲观的'], 
            weights=[0.4, 0.1, 0.5], 
            k=1)[0]
        user_belief = retry_belief_conversion(agent, row,attitude)
        print('1')
        chunk_copy.at[idx, 'belief'] = user_belief
        chunk_copy.at[idx, 'attitude'] = attitude
    return chunk_copy

def process_dataframe(df, agent, num_threads=96):
    """使用多线程处理数据框"""
    # 计算每个线程处理的数据量
    chunk_size = len(df) // num_threads
    if chunk_size == 0:
        chunk_size = 1
        num_threads = min(len(df), num_threads)
    
    # 将数据分成多个块
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    processed_chunks = []
    
    # 使用线程池处理数据
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_chunk, chunk, agent): chunk for chunk in chunks}
        
        with tqdm(total=len(df), desc="处理数据进度") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    processed_chunk = future.result()
                    processed_chunks.append(processed_chunk)
                    pbar.update(len(processed_chunk))
                except Exception as e:
                    print(f"处理数据块时发生错误: {str(e)}")
    
    # 合并所有处理后的数据
    return pd.concat(processed_chunks)

def save_results(df, output_dir):
    """保存分析结果"""
    save_dir = f"{output_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存详细结果
    result_df = df[['user_id', 'belief','attitude']]
    length=len(result_df)
    result_path = os.path.join(save_dir, f'belief_{length}_0129.csv')
    result_df.to_csv(result_path, index=False)
    
    print(f"结果已保存至: {result_path}")
    
    return result_df

def init_belief(db_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/data/UserDB/sys_100_ablation_all.db', 
                table_name='Profiles'):
    # 初始化Agent
    agent = BaseAgent(config_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/config_random/deepseek_yyz.yaml')
        
    try:
        # 读取数据库
        print(f"正在从数据库读取 {table_name} 的数据...")
        df = read_from_db(db_path, table_name)

        # df=df.head(50)
        print(f'{len(df)}条数据')
        print(df.head(5))
        # 处理数据
        print(f"开始处理数据，共 {len(df)} 条记录...")
        processed_df = process_dataframe(df, agent)
        
        # 保存结果
        avg_score = save_results(processed_df, '/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/util/belief')  # 使用结束日期作为保存目录名
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        raise

init_belief(db_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/logs_500_4o_mini/sys_500.db')
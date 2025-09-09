import pandas as pd
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent.futures
import time
import sys

# 将父目录添加到 sys.path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用相对导入
from Agent2 import BaseAgent  # 直接导入

SYSTEM_PROMPT = '''## 你是一个信念分析专家。请分析以下文本中关于市场的5个关键指标,并给出0-10的评分，其中5分代表中性：

- 市场趋势: 0表示极度下跌,10表示极度上涨
- 市场估值: 0表示极度高估,10表示极度低估  
- 经济状况: 0表示极度恶化,10表示极度向好
- 市场情绪: 0表示极度悲观,10表示极度乐观
- 自我评价: 0表示极度不自信,10表示极度自信

## 请按以下JSON格式返回结果，并且按照以下格式返回：
{
    "市场趋势": 分数,
    "市场估值": 分数,
    "经济状况": 分数,
    "市场情绪": 分数,
    "自我评价": 分数
}
'''

def read_from_db(db_path, table_name, start_date, end_date):
    """从SQLite数据库读取指定日期范围的数据
    
    Args:
        db_path (str): 数据库文件路径
        table_name (str): 表名
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
    """
    try:
        conn = sqlite3.connect(db_path)
        
        query = f"""
            SELECT * FROM {table_name}
            WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
        """
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            print(f"警告：在 {start_date} 到 {end_date} 期间没有找到任何数据")
        else:
            print(f"成功读取 {start_date} 到 {end_date} 的数据，共 {len(df)} 条记录")
            
        return df
    except Exception as e:
        print(f"读取数据库时发生错误: {e}")
        raise

def retry_float_conversion(agent, text, max_retries=3, delay=1):
    """带重试机制的市场指标分析函数"""
    for attempt in range(max_retries):
        try:
            response = agent.get_response(user_input=f'{SYSTEM_PROMPT}\n{str(text)}',response_format={"type": "json_object"}).get("response")
            # 解析JSON响应
            import json
            scores = json.loads(response)
            print(scores)
            return scores
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"分析文本失败 '{text}': {e}")
                return { 
                    "市场趋势": 5.0,
                    "市场估值": 5.0,
                    "经济状况": 5.0,
                    "市场情绪": 5.0,
                    "自我评价": 5.0
                }
            time.sleep(delay)
    return {
        "市场趋势": 5.0,
        "市场估值": 5.0,
        "经济状况": 5.0,
        "市场情绪": 5.0,
        "自我评价": 5.0
    }

def process_chunk(chunk, agent):
    """处理数据块的函数"""
    chunk_copy = chunk.copy()
    # 为每个指标创建新列
    indicators = ["市场趋势", "市场估值", "经济状况", "市场情绪", "自我评价"]
    for indicator in indicators:
        chunk_copy[indicator] = 5.0  # 默认值
    
    # 处理每行数据
    for idx, row in chunk_copy.iterrows():
        if pd.notna(row['belief']):
            scores = retry_float_conversion(agent, row['belief'])
            for indicator in indicators:
                chunk_copy.at[idx, indicator] = scores[indicator]
    
    return chunk_copy

def process_dataframe(df, agent, num_threads=100):
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
    save_dir = f"{output_dir}/ablation_news/rumor"
    os.makedirs(save_dir, exist_ok=True)
    
    # 按日期分组计算平均得分
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    indicators = ["市场趋势", "市场估值", "经济状况", "市场情绪", "自我评价"]
    
    # 计算每个指标的每日平均分
    daily_scores = df.groupby('date')[indicators].mean()
    
    # 保存详细结果
    result_df = df[['user_id', 'belief'] + indicators + ['date']]
    result_path = os.path.join(save_dir, 'market_indicators.csv')
    result_df.to_csv(result_path, index=False)
    
    # 保存每日平均得分
    with open(os.path.join(save_dir, 'daily_average_indicators.txt'), 'w') as f:
        f.write("Date\t" + "\t".join(indicators) + "\n")
        for date, row in daily_scores.iterrows():
            f.write(f"{date}\t" + "\t".join([f"{score:.4f}" for score in row]) + "\n")
    
    print(f"结果已保存至: {save_dir}")
    print("每日平均指标得分:")
    print(daily_scores)
    
    return daily_scores.mean().to_dict()  # 返回整个时期的平均得分

def calculate_belief(db_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/tmp/Forum_DB_test/sample.db', 
                    table_name='posts', 
                    start_date='2023-06-15',
                    end_date='2023-06-16'):
    # 初始化Agent
    agent = BaseAgent(config_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/config_random/gaochao_4o.yaml')
        
    try:
        # 读取数据库
        print(f"正在从数据库读取 {start_date} 到 {end_date} 的数据...")
        df = read_from_db(db_path, table_name, start_date, end_date)
        
        # 提前处理
        df = df[df['type']!='repost']
        df = df.reset_index(drop=True)
        # df=df.head(50)
        print(f'{len(df)}条数据')

        # 处理数据
        print(f"开始处理数据，共 {len(df)} 条记录...")
        processed_df = process_dataframe(df, agent)
        
        # 保存结果
        avg_score = save_results(processed_df, '/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/vis')  # 使用结束日期作为保存目录名
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        raise

calculate_belief(db_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/logs_100_0128_4o-mini_rumor/forum_100.db', 
                    table_name='posts', 
                    start_date='2023-06-15',
                    end_date='2023-07-15')
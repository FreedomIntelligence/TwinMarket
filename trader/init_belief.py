# -*- coding: utf-8 -*-
"""
用户投资信念初始化模块

该模块负责为交易系统中的用户生成初始的投资信念值，主要功能包括：
- 从用户数据库读取用户特征信息
- 基于用户特征生成个性化的投资信念
- 使用AI代理分析用户投资心理和市场观点
- 支持多线程并发处理大量用户数据
- 自动保存生成的信念值数据

核心功能：
- 用户特征分析：处置效应、彩票偏好、分散投资等心理特征
- 市场观点生成：基于用户特征生成对市场的看法
- 投资态度分类：乐观、中性、悲观三种态度随机分配
- 批量处理：支持大规模用户数据的并发处理

适用场景：
- 交易系统初始化
- 用户投资心理建模
- 市场情绪分析
- 投资策略个性化
"""

# 标准库导入
import concurrent.futures
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# 第三方库导入
import pandas as pd
from tqdm import tqdm

# 本地模块导入
# 添加根目录到Python路径，以便导入Agent模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agent import BaseAgent  # 修复：使用正确的Agent模块路径 



# ============================ 全局配置常量 ============================

# 默认系统投资信念模板
# 当AI生成信念失败时使用的备用信念文本，包含了典型投资者的市场观点
SYSTEM_PROMPT = (
    '当前时间点下，我认为未来1个月市场将呈现震荡调整趋势。'
    '从历史规律来看，市场在经历一段时间的上涨后，往往会出现回调或盘整，'
    '因此短期内需警惕可能的下跌风险。同时，市场也可能在关键支撑位附近企稳，'
    '形成新的上涨动力。从市场估值来看，我认为当前市场整体估值处于合理区间，'
    '但部分热门板块可能存在高估风险，需谨慎对待。对于宏观经济走势，我持中性态度，'
    '虽然经济复苏仍在持续，但通胀压力和货币政策的不确定性可能对市场造成一定压力。'
    '市场情绪方面，我认为当前市场情绪中性偏谨慎，投资者在乐观与悲观之间摇摆，'
    '情绪波动较大，需警惕"追涨杀跌"的心理陷阱。结合我的历史交易表现和投资风格，'
    '我认为自己的投资水平处于中等水平，能够通过技术分析和基本面研究捕捉部分机会，'
    '但情绪驱动的决策模式和较高的处置效应（如"过早卖出盈利股票，长期持有亏损股票"）'
    '限制了整体收益的提升。记住，市场总是周期波动的，涨多了会跌，跌多了会涨，'
    '保持冷静和耐心是关键。'
)

def read_from_db(db_path, table_name):
    """
    从SQLite数据库读取用户特征数据
    
    该函数从指定的数据库表中读取特定日期的用户特征数据，
    用于后续的投资信念生成。固定查询2023-06-14的数据作为基准。
    
    Args:
        db_path (str): SQLite数据库文件路径
        table_name (str): 要查询的数据表名称
        
    Returns:
        pd.DataFrame: 包含用户特征信息的DataFrame
        
    Raises:
        Exception: 数据库连接或查询失败时抛出异常
        
    Note:
        - 固定查询日期为2023-06-14 00:00:00
        - 会自动检查查询结果是否为空
        - 包含详细的日志输出用于调试
    """
    try:
        # 建立数据库连接
        conn = sqlite3.connect(db_path)
        current_time = '2023-06-14 00:00:00'  # 固定查询基准日期
        
        # 构建SQL查询语句，获取指定日期的用户数据
        query = f"SELECT * FROM {table_name} WHERE created_at ='{current_time}'"
            
        # 执行查询并转换为DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 检查查询结果并输出日志
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
    根据用户特征和投资态度生成个性化的信念生成提示词
    
    该函数是信念生成的核心，它将用户的各种投资特征（心理特征、历史表现、
    投资偏好等）整合成一个完整的提示词，用于AI代理生成个性化的投资信念。
    
    提示词包含的信息：
    - 投资风格和策略类型
    - 心理行为特征（处置效应、彩票偏好、分散投资）
    - 历史投资表现数据
    - 用户自我描述
    - 市场一般规律参考
    - 信念生成指导要求
    
    Args:
        row (pd.Series): 用户信息数据行，包含所有用户特征字段
        attitude (str): 用户投资态度，如"乐观的"、"悲观的"、"对市场态度中性的"
        
    Returns:
        str: 格式化的AI提示词，用于生成个性化投资信念
        
    Note:
        - 提示词采用第一人称视角，增强个性化效果
        - 包含了完整的投资心理学要素
        - 提供了市场规律作为分析参考框架
    """
    # ============================ 用户特征数据解析 ============================
    # 基础投资特征
    strategy = row['strategy']                                    # 投资策略类型
    disposition_effect = row['bh_disposition_effect_category']    # 处置效应水平
    lottery_preference = row['bh_lottery_preference_category']    # 彩票偏好程度
    diversification = row['bh_underdiversification_category']     # 投资分散化程度
    
    # 投资表现数据
    total_return = row['total_return']                            # 总投资回报金额
    return_rate = row['return_rate']                              # 投资回报率
    stock_returns = row['stock_returns']                          # 持仓股票详细表现
    
    # 投资偏好和描述
    fol_ind = row['fol_ind']                                      # 关注的行业领域
    self_description = row['self_description']                    # 用户自我描述

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
    """
    带重试机制的投资信念生成函数
    
    该函数使用AI代理为单个用户生成投资信念，包含了完善的重试机制
    以确保生成的可靠性。如果AI生成失败，会使用默认的系统信念。
    
    处理流程：
    1. 根据用户特征生成提示词
    2. 调用AI代理生成投资信念
    3. 如果失败则重试，最多重试指定次数
    4. 最终失败时返回默认系统信念
    
    Args:
        agent (BaseAgent): AI代理实例
        row (pd.Series): 用户特征数据行
        attitude (str): 用户投资态度
        max_retries (int): 最大重试次数，默认3次
        delay (int): 重试间隔时间（秒），默认1秒
        
    Returns:
        str: 生成的投资信念文本
        
    Note:
        - 包含完善的异常处理和重试机制
        - 失败时会使用SYSTEM_PROMPT作为备用信念
        - 会输出生成的信念文本用于调试
    """
    for attempt in range(max_retries):
        try:
            # 生成个性化提示词
            prompt = get_init_prompt(row, attitude)
            # 调用AI代理生成投资信念
            response = agent.get_response(user_input=prompt).get("response")
            # 输出生成的信念内容用于调试
            print(response)
            return str(response)
        except Exception as e:
            # 如果是最后一次尝试，使用默认信念
            if attempt == max_retries - 1:
                print(f"用户 '{row['user_id']}' 信念生成失败: {e}")
                response = SYSTEM_PROMPT
                return response
            # 等待后重试
            time.sleep(delay)
    # 所有重试都失败，返回默认系统信念
    return SYSTEM_PROMPT
     

def process_chunk(chunk, agent):
    """
    处理单个数据块的信念生成函数
    
    该函数处理一个数据块中的所有用户，为每个用户生成投资信念和态度。
    这是多线程处理的基本单元，每个线程会调用此函数处理一部分用户数据。
    
    处理流程：
    1. 复制数据块避免修改原始数据
    2. 为每个用户随机分配投资态度
    3. 生成个性化的投资信念
    4. 将结果添加到数据中
    
    Args:
        chunk (pd.DataFrame): 要处理的用户数据块
        agent (BaseAgent): AI代理实例
        
    Returns:
        pd.DataFrame: 处理后的数据块，包含新增的belief和attitude列
        
    Note:
        - 投资态度按40%乐观、10%中性、50%悲观的比例随机分配
        - 每个用户的信念生成都包含重试机制
    """
    # 复制数据块，避免修改原始数据
    chunk_copy = chunk.copy()
    
    # ============================ 初始化新列 ============================
    chunk_copy['belief'] = ''      # 投资信念列
    chunk_copy['attitude'] = ''    # 投资态度列
    
    # ============================ 逐行处理用户数据 ============================
    for idx, row in chunk_copy.iterrows():
        # 随机分配投资态度（乐观40%、中性10%、悲观50%）
        attitude = random.choices(
            ['乐观的', '对市场态度中性的', '悲观的'], 
            weights=[0.4, 0.1, 0.5], 
            k=1
        )[0]
        
        # 生成用户的投资信念
        user_belief = retry_belief_conversion(agent, row, attitude)
        print('1')  # 进度指示
        
        # 保存生成结果
        chunk_copy.at[idx, 'belief'] = user_belief
        chunk_copy.at[idx, 'attitude'] = attitude
        
    return chunk_copy

def process_dataframe(df, agent, num_threads=96):
    """
    使用多线程并发处理大规模用户数据
    
    该函数将大型DataFrame分割成多个数据块，使用线程池并发处理，
    显著提高大规模用户信念生成的处理效率。
    
    处理策略：
    1. 根据线程数计算最优数据块大小
    2. 将DataFrame分割成多个数据块
    3. 使用线程池并发处理各个数据块
    4. 实时显示处理进度
    5. 合并所有处理结果
    
    Args:
        df (pd.DataFrame): 要处理的完整用户数据
        agent (BaseAgent): AI代理实例
        num_threads (int): 并发线程数，默认96个线程
        
    Returns:
        pd.DataFrame: 处理完成的完整数据，包含所有用户的信念和态度
        
    Note:
        - 自动调整线程数以适应数据量
        - 包含完善的异常处理机制
        - 使用tqdm显示实时处理进度
        - 线程数可根据系统性能调整
    """
    # ============================ 数据分块策略 ============================
    # 计算每个线程处理的数据量
    chunk_size = len(df) // num_threads
    if chunk_size == 0:
        chunk_size = 1
        num_threads = min(len(df), num_threads)
    
    # 将数据分成多个块，便于并发处理
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    processed_chunks = []
    
    # ============================ 多线程并发处理 ============================
    # 使用线程池执行并发处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有数据块处理任务
        futures = {executor.submit(process_chunk, chunk, agent): chunk for chunk in chunks}
        
        # 显示处理进度并收集结果
        with tqdm(total=len(df), desc="处理数据进度") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    # 获取处理结果
                    processed_chunk = future.result()
                    processed_chunks.append(processed_chunk)
                    # 更新进度条
                    pbar.update(len(processed_chunk))
                except Exception as e:
                    print(f"处理数据块时发生错误: {str(e)}")
    
    # ============================ 结果合并 ============================
    # 合并所有处理后的数据块
    return pd.concat(processed_chunks)

def save_results(df, output_dir):
    """
    保存投资信念生成结果到CSV文件
    
    该函数将处理完成的用户投资信念数据保存到指定目录，
    只保存必要的字段（用户ID、信念、态度），便于后续使用。
    
    Args:
        df (pd.DataFrame): 包含完整用户信息和生成结果的DataFrame
        output_dir (str): 输出目录路径
        
    Returns:
        pd.DataFrame: 保存的结果DataFrame，只包含核心字段
        
    Note:
        - 文件名格式为 belief_{用户数量}_0129.csv
        - 只保存user_id、belief、attitude三个关键字段
        - 会自动创建输出目录
    """
    # 确保输出目录存在
    save_dir = f"{output_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取并保存核心结果字段
    result_df = df[['user_id', 'belief', 'attitude']]
    length = len(result_df)
    result_path = os.path.join(save_dir, f'belief_{length}_0129.csv')
    result_df.to_csv(result_path, index=False)
    
    print(f"结果已保存至: {result_path}")
    
    return result_df

def init_belief(
    db_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/data/UserDB/sys_100_ablation_all.db', 
    table_name='Profiles'
):
    """
    用户投资信念初始化主函数
    
    该函数是整个信念初始化流程的主入口，负责协调所有子模块完成
    大规模用户投资信念的批量生成和保存。
    
    主要流程：
    1. 初始化AI代理
    2. 从数据库读取用户特征数据
    3. 使用多线程并发处理所有用户
    4. 保存生成的信念数据到文件
    
    Args:
        db_path (str): 用户数据库文件路径，默认使用系统配置路径
        table_name (str): 数据表名称，默认为'Profiles'
        
    Raises:
        Exception: 当处理过程中发生任何错误时抛出异常
        
    Note:
        - 使用固定的AI配置文件路径
        - 结果保存到固定的belief目录
        - 包含完整的错误处理和日志输出
    """
    # ============================ AI代理初始化 ============================
    # 初始化AI代理，使用指定的配置文件
    agent = BaseAgent(
        config_path='./config_random/deepseek_yyz.yaml'
    )
        
    try:
        # ============================ 数据读取 ============================
        print(f"正在从数据库读取 {table_name} 的数据...")
        df = read_from_db(db_path, table_name)

        # 可选：限制处理数据量用于测试
        # df = df.head(50)
        
        print(f'共{len(df)}条数据')
        print(df.head(5))  # 显示前5条数据预览
        
        # ============================ 批量处理 ============================
        print(f"开始处理数据，共 {len(df)} 条记录...")
        processed_df = process_dataframe(df, agent)
        
        # ============================ 结果保存 ============================
        # 保存处理结果到指定目录
        save_results(
            processed_df, 
            './util/belief'
        )
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        raise

# ============================ 主程序执行 ============================
# 执行信念初始化，使用指定的数据库路径
if __name__ == "__main__":
    # 使用500用户的数据库进行信念初始化
    init_belief(
        db_path='./logs_500_4o_mini/sys_500.db'
    )
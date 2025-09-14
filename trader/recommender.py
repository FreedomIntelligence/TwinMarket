"""
股票推荐系统模块

该模块实现了基于股票关系图的智能推荐系统，主要功能包括：
- 构建股票之间的关联关系网络
- 基于用户投资组合推荐相关股票
- 缓存机制优化性能
- 数据验证和过滤

核心算法：
通过分析股票组合数据，构建股票之间的关联关系图，
当用户持有某些股票时，系统会推荐与这些股票相关的其他股票。

适用场景：
- 投资组合多样化建议
- 相关股票发现
- 风险分散策略支持
"""

# 标准库导入
import os
import pickle
from collections import defaultdict

# 第三方库导入
import pandas as pd


class StockRecommender:
    """
    股票推荐系统类
    
    基于股票关联关系的智能推荐系统，通过分析股票组合数据构建关系网络，
    为用户提供个性化的股票投资建议。
    
    核心功能：
    - 股票关系图构建：分析股票组合数据，建立股票间的关联关系
    - 智能推荐算法：基于用户当前持仓推荐相关股票
    - 缓存优化：使用pickle缓存关系图，提高系统性能
    - 数据验证：确保推荐的股票都是有效可交易的
    
    算法原理：
    如果两只股票经常出现在同一个投资组合中，则认为它们之间存在关联关系。
    基于这种关联关系，当用户持有某些股票时，系统会推荐相关的其他股票。
    
    Attributes:
        file_path (str): 股票组合数据文件路径
        cache_dir (str): 缓存文件目录
        stock_path (str): 有效股票列表文件路径
        stock_relations (dict): 股票关系图
        valid_stocks (list): 有效股票代码列表
        
    Example:
        >>> recommender = StockRecommender()
        >>> recommendations = recommender.recommend_portfolio(['000001', '000002'], top_n=3)
        >>> print(recommendations)  # 返回推荐的股票列表
    """
    def __init__(self,
                 file_path: str = 'data/guba_data/guba_data.csv',
                 cache_dir: str = 'trader/cache',
                 stock_path: str = 'data/UserDB/userdata/stock_profile.csv'
                 ):
        """
        初始化股票推荐系统
        
        Args:
            file_path (str): 股票组合数据文件路径，包含股票分组信息
            cache_dir (str): 缓存目录，用于存储构建的关系图
            stock_path (str): 有效股票列表文件路径，用于验证股票代码
        """
        # ============================ 路径配置 ============================
        self.file_path = file_path      # 股票组合数据文件路径
        self.cache_dir = cache_dir      # 缓存文件目录
        self.stock_path = stock_path    # 有效股票列表文件路径
        
        # ============================ 核心数据初始化 ============================
        # 加载或构建股票关系图（优先使用缓存）
        self.stock_relations = self._load_or_build_stock_relations()
        # 获取有效股票代码列表，用于推荐结果验证
        self.valid_stocks = self._get_valid_stocks()

    def _get_valid_stocks(self):
        """
        获取系统中所有有效的股票代码列表
        
        从股票资料文件中读取所有有效的股票代码，用于后续的推荐结果验证。
        确保推荐的股票都是系统中可交易的有效股票。
        
        Returns:
            list: 有效股票代码列表，已去重和去空值
            
        Note:
            - 会自动过滤空值和重复项
            - 股票代码来源于stock_profile.csv文件的stock_id列
        """
        df = pd.read_csv(self.stock_path)
        return df['stock_id'].dropna().unique().tolist()  # 获取所有有效的股票代码

    def _load_or_build_stock_relations(self):
        """
        加载或构建股票关系图
        
        优先从缓存文件加载已构建的股票关系图，如果缓存不存在则重新构建。
        使用缓存机制可以显著提高系统启动速度。
        
        Returns:
            dict: 股票关系图字典，格式为 {stock_code: [related_stocks]}
            
        Note:
            - 缓存文件使用pickle格式存储
            - 如果缓存目录不存在会自动创建
            - 构建过程可能耗时较长，建议使用缓存
        """
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "stock_relations.pkl")
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            # print("加载缓存中的股票关系图...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
                
        # 缓存不存在，重新构建并保存
        # print("构建股票关系图并保存到缓存...")
        stock_relations = self._build_stock_relations()
        with open(cache_file, "wb") as f:
            pickle.dump(stock_relations, f)
        return stock_relations

    def _build_stock_relations(self):
        """
        构建股票关联关系图
        
        通过分析股票组合数据，构建股票之间的关联关系网络。
        如果两只股票经常出现在同一个投资组合中，则认为它们之间存在关联关系。
        
        算法流程：
        1. 加载股票组合数据
        2. 过滤无效股票代码
        3. 按组合名称分组股票
        4. 构建股票间的双向关联关系
        
        Returns:
            defaultdict: 股票关系图，格式为 {stock_code: [related_stocks]}
            
        Raises:
            ValueError: 当过滤后数据为空时抛出异常
            
        Note:
            - 关系是双向的：如果A关联B，则B也关联A
            - 只处理有效股票代码，确保推荐结果的可用性
        """
        df = pd.read_csv(self.stock_path)
        
        # 获取有效股票列表并过滤数据
        valid_stocks = self._get_valid_stocks()
        df = df[df['stkcd'].isin(valid_stocks)]  # 只保留有效的股票代码

        # 验证过滤后的数据
        if len(df) == 0:
            raise ValueError("过滤后数据为空，请检查股票代码前缀是否正确。")

        # 按组合名称分组股票
        group_to_stocks = defaultdict(list)
        for _, row in df.iterrows():
            group_to_stocks[row['组合名称']].append(row['stkcd'])

        # 构建股票关系图 - 同组合内的股票互相关联
        stock_relations = defaultdict(list)
        for stocks in group_to_stocks.values():
            # 为组合内每对股票建立双向关联关系
            for i in range(len(stocks)):
                for j in range(i + 1, len(stocks)):
                    stock_relations[stocks[i]].append(stocks[j])
                    stock_relations[stocks[j]].append(stocks[i])

        return stock_relations

    def recommend_portfolio(self, input_portfolio: list, top_n: int = 3) -> list:
        """
        基于用户投资组合推荐相关股票
        
        根据用户当前持有的股票，利用股票关系图推荐相关的股票。
        推荐逻辑基于股票的历史共现关系，即经常一起出现在投资组合中的股票。
        
        算法流程：
        1. 遍历用户持有的每只股票
        2. 查找与这些股票相关的所有股票
        3. 排除用户已持有的股票
        4. 验证推荐股票的有效性
        5. 返回前N个推荐结果
        
        Args:
            input_portfolio (list): 用户当前持有的股票代码列表
            top_n (int): 返回的推荐股票数量，默认为3
            
        Returns:
            list: 推荐的股票代码列表，最多返回top_n个
            
        Note:
            - 不会推荐用户已持有的股票
            - 推荐结果会验证有效性，确保可交易
            - 如果关联股票不足top_n个，返回所有可用的关联股票
        """
        # 收集所有相关股票
        related_stocks = set()
        for stock in input_portfolio:
            if stock in self.stock_relations:
                related_stocks.update(self.stock_relations[stock])
                
        # 排除用户已持有的股票
        related_stocks = related_stocks - set(input_portfolio)

        # 过滤推荐结果，确保都是有效股票
        related_stocks = [stock for stock in related_stocks if stock in self.valid_stocks]

        # 返回前N个推荐结果
        return list(related_stocks)[:top_n]
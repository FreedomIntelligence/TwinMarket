import pandas as pd
import os
import pickle
from collections import defaultdict


class StockRecommender:
    def __init__(self,
                 file_path: str = 'data/guba_data/guba_data.csv',
                 cache_dir: str = 'trader/cache',
                 stock_path: str = 'data/UserDB/userdata/stock_profile.csv'
                 ):
        self.file_path = file_path
        self.cache_dir = cache_dir
        self.stock_path = stock_path
        self.stock_relations = self._load_or_build_stock_relations()
        self.valid_stocks = self._get_valid_stocks()  # fix: 获取有效股票列表

    def _get_valid_stocks(self):
        """获取有效的股票代码列表"""
        df = pd.read_csv(self.stock_path)
        return df['stock_id'].dropna().unique().tolist()  # 获取所有有效的股票代码

    def _load_or_build_stock_relations(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "stock_relations.pkl")
        if os.path.exists(cache_file):
            # print("加载缓存中的股票关系图...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        # print("构建股票关系图并保存到缓存...")
        stock_relations = self._build_stock_relations()
        with open(cache_file, "wb") as f:
            pickle.dump(stock_relations, f)
        return stock_relations

    def _build_stock_relations(self):
        df = pd.read_csv(self.stock_path)
        # fix: 过滤数据，只保留 stock_id 列中存在的股票代码
        valid_stocks = self._get_valid_stocks()  # 使用有效股票列表
        df = df[df['stkcd'].isin(valid_stocks)]  # 只保留 stkcd 列中存在于 stock_id 列的股票代码

        if len(df) == 0:
            raise ValueError("过滤后数据为空，请检查股票代码前缀是否正确。")

        group_to_stocks = defaultdict(list)
        for _, row in df.iterrows():
            group_to_stocks[row['组合名称']].append(row['stkcd'])

        # 构建股票关系图
        stock_relations = defaultdict(list)
        for stocks in group_to_stocks.values():
            for i in range(len(stocks)):
                for j in range(i + 1, len(stocks)):
                    stock_relations[stocks[i]].append(stocks[j])
                    stock_relations[stocks[j]].append(stocks[i])

        return stock_relations

    def recommend_portfolio(self, input_portfolio: list, top_n: int = 3) -> list:
        related_stocks = set()
        for stock in input_portfolio:
            if stock in self.stock_relations:
                related_stocks.update(self.stock_relations[stock])
        related_stocks = related_stocks - set(input_portfolio)

        # fix: 过滤推荐的股票，确保它们在有效股票列表中
        related_stocks = [stock for stock in related_stocks if stock in self.valid_stocks]

        return list(related_stocks)[:top_n]


# # test
input_portfolio = ['SH688256', 'SH600900', 'SH605028']

STOCK_REC = StockRecommender()
STOCK_REC._load_or_build_stock_relations()
rec_stock = STOCK_REC.recommend_portfolio(input_portfolio=input_portfolio, top_n=3)

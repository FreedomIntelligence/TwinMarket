"""
个性化股票交易代理模块

该模块实现了一个完整的个性化股票交易代理系统，包括：
- 智能交易决策制定
- 论坛互动和社交功能
- 新闻信息检索和分析
- 技术指标分析
- 风险管理和仓位管理

功能特点：
1. 非拆单交易机制
2. 支持10个行业指数交易
3. 智能指数推荐系统
4. 多维度信息整合分析

更新记录：
- 推荐两只指数
- 优化潜在交易指数选择逻辑
"""

# 标准库导入
import asyncio
import copy
import json
import math
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Literal

# 第三方库导入
import numpy as np
import pandas as pd
import yaml
from openai import OpenAI

# 本地模块导入
from util.InformationDB import InformationDB
from util.UserDB import *
from util.ForumDB import *
from .utility import *
from .prompts import TradingPrompt
from .recommender import StockRecommender
from .IndustryDict import *

# ============================ 全局配置和数据库实例 ============================

# 信息数据库实例 - 用于新闻信息检索和分析


# 当前配置：使用配置文件管理
INFORMATION_DB = InformationDB(
    config_path="config/embedding.yaml",  # 嵌入模型配置文件路径
    database_dir="data/InformationDB_news_2023",  # 新闻数据库目录
)
INFORMATION_DB.load_database()  # 加载新闻数据库

# 股票推荐系统（暂时未使用）
# STOCK_REC = StockRecommender()
# STOCK_REC._load_or_build_stock_relations()

# 股票数据表名称
STOCK_DB_NAME = "StockData"


class PersonalizedStockTrader:
    """
    个性化股票交易代理类

    该类实现了一个完整的个性化股票交易代理，能够根据用户特征、市场信息、
    社交网络等多种因素进行智能交易决策。主要功能包括：

    核心功能：
    - 智能交易决策制定
    - 多维度信息整合分析
    - 风险管理和仓位控制
    - 论坛互动和社交功能
    - 新闻信息检索和分析
    - 技术指标分析

    交易特性：
    - 支持10个行业指数交易
    - 非拆单交易机制
    - 智能指数推荐
    - 个性化策略执行

    数据来源：
    - 用户历史交易数据
    - 实时市场数据
    - 新闻和公告信息
    - 论坛社交数据
    - 技术指标数据

    Attributes:
        user_profile (dict): 用户资料信息
        user_graph (nx.Graph): 用户社交关系图
        df_stock (pd.DataFrame): 股票数据
        forum_db_path (str): 论坛数据库路径
        user_db_path (str): 用户数据库路径
        import_news (list): 导入的新闻列表
        user_strategy (str): 用户交易策略
        is_trading_day (bool): 是否为交易日
        is_top_user (bool): 是否为顶级用户
        log_dir (str): 日志目录
        is_random_trader (bool): 是否为随机交易者
        config_path (str): 配置文件路径
        is_activate_user (bool): 用户是否激活
        belief (str): 用户信念值
    """

    def __init__(
        self,
        user_profile: dict,
        user_graph: nx.Graph,
        df_stock: pd.DataFrame,
        forum_db_path: str = None,
        user_db_path: str = None,
        import_news: list = None,
        user_strategy: str = None,
        is_trading_day: bool = True,
        is_top_user: bool = True,
        log_dir: str = "logs",
        is_random_trader: bool = False,
        config_path: str = None,
        is_activate_user: bool = True,
        belief: str = None,
    ):
        """
        初始化个性化股票交易代理

        Args:
            user_profile (dict): 用户资料信息，包含交易历史、偏好等
            user_graph (nx.Graph): 用户社交关系网络图
            df_stock (pd.DataFrame): 股票市场数据DataFrame
            forum_db_path (str, optional): 论坛数据库文件路径
            user_db_path (str, optional): 用户数据库文件路径
            import_news (list, optional): 导入的新闻信息列表
            user_strategy (str, optional): 用户交易策略类型
            is_trading_day (bool): 当前是否为交易日，默认True
            is_top_user (bool): 用户是否为顶级用户，默认True
            log_dir (str): 日志文件保存目录，默认"logs"
            is_random_trader (bool): 是否为随机交易者，默认False
            config_path (str, optional): API配置文件路径
            is_activate_user (bool): 用户是否激活，默认True
            belief (str, optional): 用户当前信念值
        """
        # ============================ 基础配置初始化 ============================
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 用户状态配置
        self.is_activate_user = is_activate_user  # 用户是否激活
        self.belief = belief  # 用户信念值

        # ============================ 核心数据初始化 ============================
        # 用户相关数据
        self.user_profile = user_profile  # 用户资料信息
        self.user_strategy = user_strategy  # 用户交易策略
        self.user_graph = user_graph  # 用户社交关系图

        # 数据库和外部服务
        self.InformationDataBase = INFORMATION_DB  # 信息数据库实例
        self.forum_db_path = forum_db_path  # 论坛数据库路径
        self.user_db_path = user_db_path  # 用户数据库路径

        # ============================ 交易数据初始化 ============================
        # 股票相关列表
        self.potential_stock_list = []  # 潜在交易股票列表
        self.all_stock_list = []  # 所有股票列表
        self.stocks_to_deal = []  # 待处理股票列表
        self.df_stock = df_stock  # 股票数据DataFrame

        # ============================ 市场信息初始化 ============================
        # 市场和新闻信息
        self.import_news = import_news  # 导入的新闻列表
        self.is_trading_day = is_trading_day  # 是否为交易日
        self.is_top_user = is_top_user  # 是否为顶级用户

        # ============================ 交易状态初始化 ============================
        # 用户和交易状态
        self.user_id = self.user_profile["user_id"]  # 用户ID
        self.forum_args = None  # 论坛交互参数
        self.decision_result = None  # 交易决策结果
        self.post_response_args = None  # 帖子回复参数
        self.stock_profile_dict = STOCK_PROFILE_DICT  # 股票资料字典
        self.conversation_history = []  # 对话历史记录
        self.system_context = None  # 系统上下文

        # ============================ AI代理初始化 ============================
        # AI相关配置
        self.is_random_trader = is_random_trader  # 是否为随机交易者
        self.config_path = config_path  # 配置文件路径
        self.base_agent = BaseAgent(config_path=self.config_path)  # 基础AI代理
        self.price_info = {}  # 价格信息缓存

    def _process_decision_result(self, decision_result: dict) -> dict:
        """
        处理交易决策结果，计算具体的交易数量和订单信息

        该方法将抽象的交易决策（如目标仓位、价格）转换为具体的交易订单，
        包括计算实际交易数量、价格限制检查、拆单逻辑等。

        Args:
            decision_result (dict): 包含股票交易决策的字典
                格式: {"stock_decisions": {stock_code: decision_info}}

        Returns:
            dict: 处理后的决策结果，包含具体的交易订单信息

        Note:
            - 所有交易数量都会调整为100股的整数倍
            - 会检查涨跌停限制
            - 卖出时会检查持仓数量限制
            - 支持拆单逻辑（目前为非拆单模式）
        """
        for stock_code, decision in decision_result.get("stock_decisions", {}).items():
            action = decision.get("action")
            if action != "hold":
                target_position = decision.get("target_position", 0)
                target_position = int(target_position * 100) / 100
                cur_position = decision.get("cur_position", 0)
                trade_position = abs(target_position - cur_position)
                target_price = decision.get("target_price", 0)

                # 获取涨停价和跌停价
                limit_up = self.price_info[stock_code].get("limit_up", float("inf"))
                limit_down = self.price_info[stock_code].get("limit_down", 0)

                # 计算交易数量
                if target_price > 0:
                    quantity = (
                        (trade_position / 100)
                        * self.user_profile["total_value"]
                        / target_price
                    )
                    if quantity < 100:
                        quantity = 100
                    else:
                        quantity = (quantity // 100) * 100  # 向下取整为100的倍数

                    # 如果是卖出操作，确保不超过当前持仓数量
                    if action == "sell" and stock_code in self.user_profile.get(
                        "cur_positions", {}
                    ):
                        current_shares = self.user_profile["cur_positions"][
                            stock_code
                        ].get("shares", 0)
                        quantity = min(quantity, current_shares)
                        # 确保卖出数量是100的倍数
                        quantity = (quantity // 100) * 100

                    # 拆单逻辑
                    if quantity >= 100:
                        decision["sub_orders"] = [
                            {"quantity": quantity, "price": float(target_price)}
                        ]
                    else:
                        decision["sub_orders"] = []

                    decision["quantity"] = int(quantity)  # 更新决策结果中的数量
            else:
                decision["quantity"] = 0
                decision["sub_orders"] = []

        return decision_result

    def get_stock_data(
        self,
        stock_codes: list,
        indicators: list,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """
        获取指定股票在指定时间范围内的技术指标数据

        该方法从股票数据库中提取指定股票的历史技术指标数据，用于技术分析和决策支持。
        支持多种技术指标的同时查询，并确保数据的时间范围不超过当前日期。

        Args:
            stock_codes (list): 股票代码列表
            indicators (list): 需要查询的技术指标列表
            start_date (str, optional): 查询开始日期，格式为'YYYY-MM-DD'
            end_date (str, optional): 查询结束日期，格式为'YYYY-MM-DD'

        Returns:
            dict: 股票数据字典，格式为：
                {
                    stock_code: {
                        'data': [时间序列数据记录列表],
                        'start_date': '实际开始日期',
                        'end_date': '实际结束日期'
                    }
                }

        Note:
            - 结束日期会自动调整为不超过当前日期前一天
            - 只返回在数据库中存在的技术指标
            - 如果查询失败会返回空字典并打印错误信息
        """
        try:
            # 首先处理日期参数

            result = {}

            # 2. 从数据库获取
            df = self.df_stock.copy(deep=True)

            for stock_code in stock_codes:

                stock_result = {}
                # 获取交易数据
                stock_data = df[df["stock_id"] == stock_code]
                trading_indicators = [ind for ind in indicators if ind in df.columns]

                if not stock_data.empty and trading_indicators:
                    # 将日期转换为datetime对象
                    current_date = pd.to_datetime(self.cur_date)
                    start = pd.to_datetime(start_date)
                    end = pd.to_datetime(end_date)

                    # 确保end_date不超过current_date前一天
                    end = min(end, current_date - pd.Timedelta(days=1))

                    period_data = stock_data[
                        (stock_data["date"] >= start) & (stock_data["date"] <= end)
                    ]

                    if not period_data.empty:
                        all_indicators = trading_indicators + ["date"]
                        period_data = period_data.assign(
                            date=period_data["date"].dt.strftime("%Y-%m-%d")
                        )
                        trading_result = {
                            "data": period_data[all_indicators].to_dict("records"),
                            "start_date": start.strftime("%Y-%m-%d"),
                            "end_date": end.strftime("%Y-%m-%d"),
                        }
                        stock_result.update(trading_result)

                result[stock_code] = stock_result

            return result

        except Exception as e:
            print(f"获取股票数据时出错: {str(e)}")
            print(
                f"参数信息: stock_codes={stock_codes}, indicators={indicators}, start_date={start_date}, end_date={end_date}"
            )
            return {}

    def _get_stock_summary(self, stock_codes: list, current_date: pd.Timestamp) -> str:
        """
        获取股票的市场表现摘要信息

        该方法为指定的股票代码生成前一交易日的市场表现摘要，
        包括价格变化、成交量、涨跌幅等关键指标。

        Args:
            stock_codes (list): 股票代码列表
            current_date (pd.Timestamp): 当前日期

        Returns:
            str: 格式化的股票市场表现摘要文本

        Note:
            - 获取前一交易日的数据
            - 如果没有数据会返回相应提示
            - 使用TradingPrompt模板格式化输出
        """
        yesterday = current_date - pd.Timedelta(days=1)
        columns = [
            "change",
            "pct_chg",
            "vol",
            "date",
            "stock_id",
            "close_price",
            "pre_close",
        ]

        df = self.df_stock.copy(deep=True)

        summary = []
        for stock_code in stock_codes:
            selected_row = df.loc[
                (df["stock_id"] == stock_code) & (df["date"] <= yesterday)
            ].sort_values("date", ascending=False)
            if not selected_row.empty:
                selected_row = selected_row.iloc[0][columns]
                stock_summary = TradingPrompt.get_stock_summary(
                    stock_code, selected_row
                )
                summary.append(stock_summary)
            else:
                summary.append(f"## 指数名称：{stock_code} 无上个交易日交易数据。")

        return "\n".join(summary)

    def _generate_initial_prompt(self, current_date: pd.Timestamp) -> str:
        formatted_date = self._format_date(current_date)
        stock_summary = self._get_stock_summary(self.stocks_to_deal, current_date)
        positions_info = self._get_stock_details(self.stocks_to_deal, type="full")

        return TradingPrompt.get_initial_prompt_fake(
            formatted_date=formatted_date,
            stocks_to_deal=self.stocks_to_deal,
            stock_summary=stock_summary,
            positions_info=positions_info,
            return_rate=self.user_profile["return_rate"],
            total_value=self.user_profile["total_value"],
            current_cash=self.user_profile["current_cash"],
            system_prompt=self.user_profile["sys_prompt"],
            user_strategy=self.user_strategy,
        )

    def _get_environment_info(
        self, current_date: pd.Timestamp, debug: bool = False
    ) -> tuple[str, bool]:
        print_debug("Getting environment information...", debug)
        news_anno = True
        if news_anno:
            new_message, queries = self._desire_agent(current_date)
            agent = self.base_agent
            input_message = [
                {
                    "role": "user",
                    "content": f"{self.system_context['content']}\n 我帮你搜索到了如下信息和公告：\n{new_message}\n请你根据你的投资风格和人设，结合你目前的持仓谈谈你的初步看法,言简意赅一些。",
                }
            ]
            self.point1 = agent.get_response(messages=input_message).get("response")
            self.conversation_history[-1][
                "content"
            ] = f"## 我想要查询的关键词如下：\n- 关键词：{queries}\n  ## 经过自行查阅相关信息，目前初步想法：\n {self.point1}"
            return new_message, True

        return None, False

    def input_info(
        self,
        stock_codes: list,
        current_date: pd.Timestamp,
        debug: bool = False,
        day_1st: bool = True,
    ) -> dict:
        """
        交易代理核心处理逻辑 - 主入口方法

        该方法是整个交易代理的核心处理流程，负责协调所有子模块完成一个完整的交易周期。
        主要处理流程包括：
        1. 论坛互动和信息收集
        2. 新闻信息检索和分析
        3. 股票推荐和筛选
        4. 技术指标数据收集
        5. 交易决策制定
        6. 结果处理和输出

        Args:
            stock_codes (list): 用户当前持有的股票代码列表
            current_date (pd.Timestamp): 当前交易日期
            debug (bool): 是否开启调试模式，默认False
            day_1st (bool): 是否为第一天（影响某些初始化逻辑），默认True

        Returns:
            tuple: 包含以下元素的元组
                - forum_args: 论坛互动参数
                - user_id: 用户ID
                - decision_result: 交易决策结果
                - post_response_args: 帖子回复参数
                - conversation_history: 对话历史记录

        Note:
            这是一个复杂的多步骤处理流程，每个步骤都有详细的时间统计和调试信息输出。
            方法会根据用户类型（普通用户/顶级用户/随机交易者）执行不同的处理逻辑。
        """
        # ============================ 基础参数设置 ============================
        self.is_day_1 = day_1st  # 是否为第一天
        self.stock_codes = stock_codes  # 用户持有股票代码
        self.cur_date = current_date.strftime("%Y-%m-%d")  # 当前日期字符串
        self.debug = debug  # 调试模式标识

        # ============================ 论坛互动初始化 ============================
        start_time = time.time()
        self.rec_post = []  # 推荐帖子列表
        self.forum_args = {}  # 论坛交互参数

        # ============================ AI对话上下文初始化 ============================
        # 构建系统提示上下文
        self.system_context = TradingPrompt.get_system_prompt_new(
            self.user_profile,
            self.user_strategy,
            self.stock_profile_dict,
            self.stock_codes,
        )

        # 构建用户初始提示和代理回复
        self.user_prompt = TradingPrompt.get_user_first_prompt(
            self.user_profile,
            self.user_strategy,
            self.stock_profile_dict,
            pd.to_datetime(self.cur_date),
            self.is_trading_day,
            self.belief,
        )
        agent_first_prompt = TradingPrompt.get_agent_first_response(
            self.user_profile,
            self.user_strategy,
            self.stock_profile_dict,
            pd.to_datetime(self.cur_date),
            self.is_trading_day,
            self.belief,
        )

        # 初始化对话历史
        self.conversation_history.append(self.system_context)
        self.conversation_history.append(self.user_prompt)
        self.conversation_history.append(agent_first_prompt)

        if not day_1st:
            # 注意：ForumDB 内部按日期字符串比较，传入 end_date 为当前日可覆盖到前一日全量
            self.rec_post = recommend_post_graph(
                target_user_id=self.user_id,
                start_date=datetime(2023, 6, 14),
                end_date=current_date,
                db_path=self.forum_db_path,
                graph=self.user_graph,
                max_return=5,
            )

            self.forum_args, self.forum_summary = self._forum_action()
            print_debug(f"self.forum_args: {self.forum_args}", debug)
            self.conversation_history.append(
                {"role": "user", "content": self.forum_summary}
            )
            # todo: add to conversation history: self.forum_summary
        print_debug(f"刷帖模块耗时: {time.time() - start_time:.2f}秒", debug)

        # fix： 不是activate user直接返回
        if self.is_activate_user:

            print_debug(
                f"User {self.user_id} is activate: {self.is_activate_user}", debug
            )

            # 全体新闻
            start_time = time.time()
            # 读取全体重要新闻广播
            if self.is_top_user:  # todo: check
                self._read_news()
                # 全体新闻逻辑
                print_debug(
                    f"全体新闻模块耗时: {time.time() - start_time:.2f}秒", debug
                )

            # 系统随机推荐股票
            if self.is_trading_day:
                start_time = time.time()
                self._get_rec_stock()
                # print_debug(f"系统随机推荐股票耗时: {time.time() - start_time:.2f}秒", debug)

            # 查找新闻（公告）
            if not self.is_random_trader:
                start_time = time.time()
                environment_info, whether_decision = self._get_environment_info(
                    current_date, debug
                )
                print_debug(
                    f"查找新闻（公告）耗时: {time.time() - start_time:.2f}秒", debug
                )

            # 更新 belief
            if not self.is_random_trader:
                start_time = time.time()
                # fix
                self.belief = self._update_belief()
                # self.belief = tmp_belief.get("belief", self.belief)
                print_debug(
                    f"更新 belief 耗时: {time.time() - start_time:.2f}秒", debug
                )

            # 选择待交易的股票 TODO：可能为空
            if self.is_trading_day and not self.is_random_trader:
                start_time = time.time()
                self.stocks_to_deal = self._choose_stocks()
                print_debug(
                    f"选择待交易的股票耗时: {time.time() - start_time:.2f}秒", debug
                )

            # 收集查询的数据 TODO：可能为空
            if self.is_trading_day and not self.is_random_trader:
                if len(self.stocks_to_deal) > 0:
                    start_time = time.time()
                    self._data_collection(debug)
                    print_debug(
                        f"收集查询的数据耗时: {time.time() - start_time:.2f}秒", debug
                    )

            # return self.user_id, self.decision_result, post_response_args
            # self.output_decision()

            if self.is_trading_day:
                start_time = time.time()
                if not self.is_random_trader:
                    if len(self.stocks_to_deal) > 0:
                        self.decision_result = self._make_final_decision(debug)
                    else:
                        self.decision_result = None
                else:
                    self.decision_result = self._make_final_decision_random(debug)
                print_debug(
                    f"{'是' if self.is_random_trader else '不是'}random trader；生成最终决策耗时: {time.time() - start_time:.2f}秒",
                    debug,
                )

            # 处理决策结果，计算每个股票的交易数量
            if self.is_trading_day:
                if self.decision_result:
                    start_time = time.time()
                    self.decision_result = self._process_decision_result(
                        self.decision_result
                    )
                else:
                    self.decision_result = {"error": "用户不选择交易任何股票"}
                print_debug(
                    f"处理决策结果耗时: {time.time() - start_time:.2f}秒", debug
                )

            # 发帖
            start_time = time.time()
            print_debug("Interacting with environment...", debug)
            self.post_response_args = self._intention_agent(
                current_date, self.conversation_history
            )  # post, type, belief
            print_debug(self.post_response_args, debug)
            print_debug(
                f"与 environment 交互耗时: {time.time() - start_time:.2f}秒", debug
            )

        else:
            self.decision_result = {"error": "用户没有被激活"}
            self.post_response_args = None

        return (
            self.forum_args,
            self.user_id,
            self.decision_result,
            self.post_response_args,
            self.conversation_history,
        )

    def _make_final_decision_random(self, debug: bool = False) -> dict:

        print_debug("Generating final decision...", debug)

        # 随机决策
        self.stocks_to_deal = list(
            set(self.stock_codes) | set(self.potential_stock_list)
        )

        # 决策-预备知识
        self.price_info = self._get_price_limits(self.stocks_to_deal)
        cur_positions = self.user_profile.get("cur_positions", {})
        position_info = {
            stock_code: {
                "current_position": cur_positions.get(stock_code, {}).get("ratio", 0)
            }
            for stock_code in self.stocks_to_deal
        }
        # 排除掉要交易的股票
        total_position = (
            sum(
                details["ratio"]
                for stock_code, details in cur_positions.items()
                if stock_code not in self.stocks_to_deal
            )
            if cur_positions
            else 0.0
        )
        available_position = 100 - total_position

        # 生成随机的决策
        decision_args = {}
        decision_args["stock_decisions"] = self._generate_random_decision()
        decision_args["stock_decisions"] = convert_values_to_float(
            decision_args["stock_decisions"]
        )
        # print_debug(f"Decision response: {decision_args}", debug)
        # 验证决策
        # decision_args = self._validate_decision(decision_args, price_info, cur_positions, available_position)
        decision_args = self._polish_decision(
            decision_args, cur_positions, available_position
        )
        analysis_result = "我今天的决策为自动化交易，在后续的belief更新中，请根据我的决策结果进行更新。"
        decision_result = TradingPrompt.decision_json_to_prompt(
            decision_args, self.potential_stock_list
        )

        # 生成一个user的对话
        self.conversation_history.append(
            {
                "role": "user",
                "content": f"""现在是做出最终交易决策的时候。请基于之前的分析，结合你的投资风格和人设，首先进行分析，然后对每支行业指数做出具体的交易决策并给出你的理由。""",
            }
        )
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": f"""{analysis_result}\n{decision_result}""",
            }
        )

        return decision_args  # 如果验证通过，返回决策

    def _generate_random_decision(
        self,
    ) -> dict:
        # 获取前一天的股票数据
        df = self.df_stock.copy(deep=True)
        yesterday = pd.to_datetime(self.cur_date) - pd.Timedelta(days=1)

        stock_decisions = {}

        for stock_code in self.stocks_to_deal:
            # 获取前一天的涨跌数据
            stock_data = df[
                (df["stock_id"] == stock_code) & (df["date"] <= yesterday)
            ].sort_values("date", ascending=False)

            if not stock_data.empty:
                pct_change = stock_data.iloc[0]["pct_chg"]  # 获取涨跌幅

                price = self.price_info[stock_code]["pre_close"]

                if stock_code in self.potential_stock_list:
                    trading_position = round(min(abs(pct_change) * 8, 30), 2)
                    action = "buy"
                else:
                    if pct_change > 0:  # 如果前一天上涨
                        # 根据涨幅大小决定买入仓位(0-20之间)
                        trading_position = round(min(abs(pct_change) * 5, 30), 2)
                        action = "buy"
                    elif pct_change < 0:  # 如果前一天下跌
                        # 根据跌幅大小决定卖出仓位(0-20之间)
                        trading_position = round(min(abs(pct_change) * 5, 30), 2)
                        action = "sell"
                    else:
                        trading_position = 0
                        action = "hold"
                        price = 0

                stock_decisions[stock_code] = {
                    "action": action,
                    "trading_position": trading_position,
                    "target_price": price,
                }

        return stock_decisions

    def _get_price_limits(self, stock_codes: list) -> dict:
        """
        获取股票的价格限制信息（涨跌停价格）

        根据前一交易日的收盘价计算当日的涨跌停价格限制，
        用于交易决策中的价格校验和风险控制。

        Args:
            stock_codes (list): 股票代码列表

        Returns:
            dict: 价格限制信息字典，格式为：
                {
                    stock_code: {
                        'pre_close': float,    # 前收盘价
                        'limit_up': float,     # 涨停价
                        'limit_down': float    # 跌停价
                    }
                }

        Raises:
            ValueError: 当无法获取股票价格数据时抛出异常

        Note:
            - 涨跌停限制为10%（1.1倍和0.9倍）
            - 基于前一交易日收盘价计算
        """
        current_date = pd.to_datetime(self.cur_date)
        df = self.df_stock.copy(deep=True)

        # 创建结果字典
        price_limits = {}

        # 筛选所有相关股票的数据
        df_filtered = df[df["stock_id"].isin(stock_codes) & (df["date"] < current_date)]

        # 对每只股票获取最新的收盘价
        for stock_code in stock_codes:
            selected_row = df_filtered[
                df_filtered["stock_id"] == stock_code
            ].sort_values("date", ascending=False)

            if not selected_row.empty:
                pre_close_price = float(selected_row.iloc[0]["close_price"])
                # 计算涨跌停价格（假设是10%的限制）
                limit_up = round(pre_close_price * 1.1, 2)
                limit_down = round(pre_close_price * 0.9, 2)

                price_limits[stock_code] = {
                    "pre_close": pre_close_price,
                    "limit_up": limit_up,
                    "limit_down": limit_down,
                }
            else:
                raise ValueError(
                    f"无法获指数 {stock_code} 在 {current_date} 的价格数据"
                )

        return price_limits

    def _format_date(self, date: pd.Timestamp) -> str:
        """
        将日期格式化为中文显示格式

        将标准的日期格式转换为中文的年月日星期格式，用于用户界面显示。

        Args:
            date (pd.Timestamp): 要格式化的日期

        Returns:
            str: 中文格式的日期字符串，如"2023年06月15日 星期四"

        Note:
            - 支持字符串类型的日期输入，会自动转换
            - 星期显示使用中文数字
        """
        weekday_map = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}

        # 如果data是str类型，转换为datetime类型
        if isinstance(date, str):
            date = pd.to_datetime(date)

        weekday = weekday_map[date.weekday()]
        return f"{date.strftime('%Y年%m月%d日')} 星期{weekday}"

    def _format_data_for_prompt(self, data: dict) -> str:
        if not data:
            return "未获取到数据"

        result = []

        # 遍历每支股票的数据
        for stock_code, stock_data in data.items():
            result.append(f"\n# 指数{stock_code} 的额外信息")
            result.append(
                f"查询区间：{stock_data.get('start_date', '')} 至 {stock_data.get('end_date', '')}"
            )

            if "data" in stock_data:
                time_series_data = stock_data["data"]

                # 获取所有非空指标名称
                valid_indicators = set()
                for record in time_series_data:
                    for k, v in record.items():
                        if k != "date" and v is not None:
                            valid_indicators.add(k)

                # 按日期显示数据
                for record in time_series_data:
                    date = record["date"]
                    result.append(f"\n## {date}")

                    for indicator in sorted(valid_indicators):
                        value = record[indicator]
                        if value is not None:  # 只显示非空值
                            mapped_ind = MAPPING_DICT.get(indicator, indicator)

                            # 根据指标类型格式化数值
                            if indicator == "elg_amount_net":
                                value_str = (
                                    f"{value:,.2f} 万元" if value else "0.00 万元"
                                )
                                trend = "净流入" if value > 0 else "净流出"
                                result.append(f"- {mapped_ind}: {value_str} ({trend})")
                            elif indicator.startswith("ma_hfq"):
                                result.append(f"- {mapped_ind}: {value:,.2f}")
                            elif indicator.startswith("macd"):
                                result.append(f"- {mapped_ind}: {value:,.3f}")
                            else:
                                result.append(f"- {mapped_ind}: {value}")

        return "\n".join(result)

    def _intention_agent(self, current_date: pd.Timestamp, conversation_history: dict):
        before_decision_history = conversation_history[:-2]
        post_agent = self.base_agent
        post_prompt = TradingPrompt.get_intention_prompt(self.belief)

        # fix
        # self.conversation_history.append({"role": "user", "content": f'''{self.user_profile['sys_prompt']}\n{post_prompt}'''})

        self.conversation_history.append({"role": "user", "content": post_prompt})
        post_response = post_agent.get_response(
            messages=self.conversation_history,
            temperature=1.3,
            # response_format={"type": "json_object"}
        )
        post_response = post_response.get("response")
        self.conversation_history.append(
            {"role": "assistant", "content": post_response}
        )
        post_response_args = parse_response_yaml(
            response=post_response,
            max_retries=3,
            prompt=f"""
                             You need to ensure the YAML keys are as follows:
                             post: ...
                             type: ...
                             belief: ...""",
        )
        # print(post_response_args)
        return post_response_args

    def _desire_agent(self, current_date: pd.Timestamp):
        """
        智能新闻查询代理 - 生成查询需求并检索相关信息

        该方法实现了一个智能的信息检索流程：
        1. 基于用户投资偏好生成查询关键词
        2. 使用AI代理分析用户的信息需求
        3. 在新闻数据库中批量检索相关信息
        4. 整合检索结果并返回格式化的新闻摘要

        处理流程：
        - 生成用户查询问题
        - AI分析并提取关键词
        - 批量检索新闻信息
        - 格式化返回结果

        Args:
            current_date (pd.Timestamp): 当前日期，用于确定新闻检索的时间范围

        Returns:
            tuple: (result_str, queries)
                - result_str (str): 格式化的新闻检索结果
                - queries (list): 生成的查询关键词列表

        Note:
            - 新闻检索范围为当前日期前7天
            - 每个查询最多返回2条最相关的新闻
            - 使用批量检索提高效率
        """
        # 第一步：生成用户的查询问题
        query_agent = self.base_agent

        self.all_stock_list = list(
            set(self.stock_codes) | set(self.potential_stock_list)
        )

        stock_details_str = self._get_stock_details(self.all_stock_list, type="basic")

        query_prompt = TradingPrompt.get_query_for_na_prompt(
            user_type=self.user_profile["user_type"],
            stock_details=stock_details_str,
            current_date=current_date.strftime("%Y年%m月%d日"),
        )

        self.conversation_history.append({"role": "user", "content": query_prompt})

        # 使用 调用异步方法
        query_response = query_agent.get_response(
            messages=self.conversation_history,
        )
        response_content = query_response.get("response")  # 获取响应内容

        self.conversation_history[-1]["content"] = (
            TradingPrompt.get_query_for_na_prompt2(
                user_type=self.user_profile["user_type"],
                stock_details=stock_details_str,
                current_date=current_date.strftime("%Y年%m月%d日"),
            )
        )

        # 提取 <output> 标签中的内容
        pattern = r"<output>(.*?)</output>"
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            query_response = match.group(1).strip()
        else:
            # 如果没有 <output> 标签，直接使用 query_response
            query_response = response_content.strip()

        self.conversation_history.append(
            {"role": "assistant", "content": f"## 我的回答如下：\n{query_response}"}
        )
        self.conversation_history.append(
            {"role": "user", "content": TradingPrompt.get_query_desire_prompt()}
        )

        # 使用 调用异步方法
        summary_response = query_agent.get_response(
            messages=self.conversation_history,
        )
        summary_response = summary_response.get("response")  # 获取响应内容

        # 删除刚刚的 user 提问
        self.conversation_history.pop()

        search_args = parse_response_yaml(response=summary_response, max_retries=3)
        queries = search_args.get("queries", None)

        # 按照重要性程度排序，只取第一个
        if queries:
            queries = queries[:1]
        else:
            return ""

        news_results_list = self.InformationDataBase.search_news_batch(
            start_date=current_date - pd.Timedelta(days=7),
            end_date=current_date,
            queries=queries,
            top_k=2,
            type=None,
        )

        result_str = ""

        # 处理批量查询结果
        for query, news_results in zip(queries, news_results_list):
            if news_results:
                samples = [a["content"] for a in news_results]
                timelines = [a["datetime"] for a in news_results]
                result_str += f"查询<{query}> 得到的新闻信息如下:\n"
                for i in range(0, len(samples)):
                    result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
                result_str += "\n"

        return result_str, queries

    def _forum_action(self) -> tuple[list[dict], str]:
        """
        处理论坛互动行为决策

        该方法负责分析推荐的论坛帖子，并决定用户对每个帖子的互动行为
        （点赞、取消点赞、转发等）。基于用户的投资风格和人设进行决策。

        处理流程：
        1. 如果是第一天，直接返回空结果
        2. 分析每个推荐帖子的内容和上下文
        3. 根据用户人设决定互动行为
        4. 生成互动行为摘要

        Returns:
            tuple[list[dict], str]: 包含两个元素的元组
                - decision_args: 论坛互动决策列表，每个元素包含：
                    {
                        "post_id": int,      # 帖子ID
                        "action": str,       # 行为类型：like/unlike/repost
                        "reason": str        # 决策理由
                    }
                - action_summary: 互动行为的文字摘要

        Note:
            - 支持对帖子进行点赞、取消点赞、转发等操作
            - 会分析帖子的引用关系和原始内容
            - 基于AI代理进行个性化决策
        """
        if self.is_day_1:
            # 首日不进行论坛互动，返回空列表
            return [], ""

        post_descriptions = []
        for post in self.rec_post:
            description = f"帖子ID: {post['id']}, 内容: {post['content']}"
            if post.get("like_score") is not None:
                description += f", 净点赞数: {post['like_score']}"
            post_descriptions.append(description)
        posts_summary = "\n".join(post_descriptions)

        # 初始化论坛消息和历史记录
        forum_message = self.conversation_history.copy()

        # 初始化决策参数列表
        decision_args = []

        # 遍历每个帖子，分别做决策
        for post in self.rec_post:
            post_id = post["id"]
            post_content = post["content"]
            post_type = post.get("type", "")

            # 获取引用的帖子内容（如果当前帖子是 repost 类型）
            reference_content = ""
            if post_type == "repost":
                reference_id = post.get("reference_id")
                if reference_id:
                    # 查询引用的帖子内容
                    with sqlite3.connect(self.forum_db_path) as conn:
                        cursor = conn.execute(
                            """
                            SELECT content FROM posts WHERE id = ?
                        """,
                            (reference_id,),
                        )
                        reference_post = cursor.fetchone()
                        if reference_post:
                            reference_content = reference_post["content"]

            # 使用 find_root_post 获取原始帖子内容
            root_post = find_root_post(post_id, self.forum_db_path)
            root_content = root_post["content"] if root_post else "未找到原始帖子"

            # 获取针对当前帖子的决策 prompt
            post_decision_prompt = f"""
            {self.user_profile['sys_prompt']}
            现在你正在浏览论坛，你需要对每个帖子做出决策，决定是否对该帖子执行操作。你的决策应该符合你的投资风格和人设。
            以下是当前帖子的信息：
            <post_id>{post_id}</post_id> 
            <content>{post_content}</content>
            """

            # 添加原始帖子的内容
            post_decision_prompt += f"""
            该帖子引用了以下内容：<ref>{root_content}</ref>
            """

            # # 如果帖子是 repost 类型，添加引用内容
            # if post_type == "repost" and reference_content:
            #     post_decision_prompt += f"""
            #     其他人对这个帖子的评论：<comment> {reference_content} </comment>
            #     """

            post_decision_prompt += f"""
            请根据以上信息决定是否对该帖子执行操作。
            你可以选择以下操作之一：
            - repost: 转发: 你认为这个帖子值得分享给更多人，可以添加你的评论
            - unlike: 取消点赞: 你认为这个帖子不值得点赞
            - like: 点赞: 你认为这是一个有价值的帖子
            
            请注意，你的分析应该是基于你看到的这些帖子的。

            请按照以下格式输出你的决策：
            <action> 操作类型 </action> <reason> 输出你的理由 </reason>
            """
            # ```yaml
            # action: <操作类型>
            # post_id: <帖子ID>
            # reason: <用一段话解释你的决策理由，不要有任何换行符和特殊符号>
            # ```

            forum_message.append({"role": "user", "content": post_decision_prompt})

            # 获取论坛行为的响应
            forum_agent = self.base_agent
            response = forum_agent.get_response(
                # user_input=post_decision_prompt,
                messages=forum_message,
                temperature=1.3,
            )
            response = response.get("response")

            # 解析 <action> 与 <reason>
            post_decision_args = {}

            reason_match = re.search(
                r"<reason>(.*?)</reason>", response, re.DOTALL | re.IGNORECASE
            )
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = response.strip()

            action_match = re.search(
                r"<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE
            )
            action_raw = action_match.group(1).strip().lower() if action_match else ""

            # 将中英文动作统一映射
            action_map = {
                "like": "like",
                "点赞": "like",
                "赞": "like",
                "喜欢": "like",
                "unlike": "unlike",
                "取消点赞": "unlike",
                "不喜欢": "unlike",
                "点踩": "unlike",
                "repost": "repost",
                "转发": "repost",
                "分享": "repost",
                "转帖": "repost",
            }

            normalized = action_map.get(action_raw)
            if not normalized:
                # 兜底：在全文中搜索关键词
                lower_resp = response.lower()
                if "<action>" in lower_resp and "</action>" in lower_resp:
                    # 动作标签存在但未匹配，视为无效
                    normalized = None
                elif any(
                    tok in lower_resp for tok in ["repost", "转发", "分享", "转帖"]
                ):
                    normalized = "repost"
                elif any(
                    tok in lower_resp
                    for tok in ["unlike", "取消点赞", "不喜欢", "点踩"]
                ):
                    normalized = "unlike"
                elif any(tok in lower_resp for tok in ["like", "点赞", "喜欢", "赞"]):
                    normalized = "like"
                else:
                    normalized = None

            post_decision_args["post_id"] = post_id
            post_decision_args["reason"] = reason
            post_decision_args["action"] = normalized

            # 统一为列表结构
            if isinstance(post_decision_args, dict):
                post_decision_args = [post_decision_args]

            # 处理转发操作：当为 repost 且尚未提供 content 时，基于原帖内容生成简短评论
            for arg in post_decision_args:
                if arg.get("action") == "repost" and not arg.get("content"):
                    target_post_id = arg.get("post_id")
                    target_post = next(
                        (
                            post
                            for post in self.rec_post
                            if post.get("id") == target_post_id
                        ),
                        None,
                    )
                    if target_post:
                        content_prompt = f"""
                        你决定转发以下帖子：
                        帖子ID: {target_post_id}
                        内容: {target_post.get("content", "")}

                        请以第一人称生成一段转发内容，简要说明你转发的原因或评论。不要换行。
                        """
                        forum_message.append(
                            {"role": "assistant", "content": content_prompt}
                        )
                        content_response = forum_agent.get_response(
                            messages=forum_message, temperature=1.0
                        )
                        content_response = content_response.get("response")
                        # 若返回为结构化字典并且包含 content，则优先使用
                        if (
                            isinstance(content_response, dict)
                            and "content" in content_response
                        ):
                            arg["content"] = str(content_response["content"])[:200]
                        else:
                            arg["content"] = (
                                str(content_response).strip().replace("\n", " ")[:200]
                            )
                        # 清理追加的prompt
                        if len(forum_message) > len(self.conversation_history):
                            forum_message.pop()

            # 将当前帖子的决策参数添加到总决策参数列表中
            decision_args.extend(post_decision_args)

            # 清理上一条帖子的内容，避免影响下一条帖子的决策
            if len(forum_message) > len(self.conversation_history):
                forum_message.pop()  # 移除上一条帖子的决策内容

        # 仅保留有效动作；为 repost 填充 content（用 reason 兜底）
        valid_actions = []
        for arg in decision_args:
            action = arg.get("action")
            if action in {"like", "unlike", "repost"}:
                if action == "repost" and not arg.get("content"):
                    arg["content"] = arg.get("reason", "")[:200]
                valid_actions.append(arg)
        decision_args = valid_actions

        # 生成行为摘要
        action_summary = f"今天是 {self._format_date(self.cur_date)}，你在论坛中看到了以下帖子：\n{posts_summary}\n\n"
        if not decision_args:
            action_summary += "你没有执行任何操作。"
        else:
            action_summary += "你执行了以下操作：\n"
            for arg in decision_args:
                action_type = arg["action"]
                post_id = arg["post_id"]
                reason = arg.get("reason", "未提供理由")
                if action_type == "repost":
                    content = arg.get("content", "")
                    action_summary += (
                        f"- 你转发了帖子 {post_id}，转发内容为：{reason}\n"
                    )
                elif action_type == "like":
                    action_summary += f"- 你点赞了帖子 {post_id}\n  理由：{reason}\n"
                elif action_type == "unlike":
                    action_summary += (
                        f"- 你取消点赞了帖子 {post_id}\n  理由：{reason}\n"
                    )

        return decision_args, action_summary

    def _get_rec_stock(self) -> list:
        """
        Recommend all stocks that are not in the current portfolio (self.stock_codes) but are present in the CSV file.

        Returns:
            list: A list of all recommended stock IDs.
        """
        df = pd.read_csv(STOCK_PROFILE_PATH2)

        # Filter out stocks that are already in the current portfolio
        available_stocks = df[~df["stock_id"].isin(self.stock_codes)]

        if available_stocks.empty:
            return

        num_recommendations = min(2, len(available_stocks))
        rec_stock = random.sample(
            available_stocks["stock_id"].tolist(), num_recommendations
        )
        # Update the potential stock list
        self.potential_stock_list = rec_stock

        return

    def _update_belief(self) -> str:
        """
        更新用户的投资信念值

        基于当前的对话历史和市场信息，使用AI代理更新用户的投资信念。
        信念值反映了用户对市场的整体看法和投资态度。

        Returns:
            str: 更新后的信念值文本

        Note:
            - 使用较高的温度参数(2.0)增加输出的多样性
            - 基于完整的对话历史进行分析
            - 信念值会影响后续的交易决策和论坛互动
        """
        pre_conversation_history = self.conversation_history.copy()
        pre_conversation_history.append(
            {
                "role": "assistant",
                "content": TradingPrompt.get_update_belief_prompt(self.belief),
            }
        )
        update_agent = self.base_agent

        response = update_agent.get_response(
            messages=pre_conversation_history, temperature=2.0
        )

        response = response.get("response")
        # fix
        # belief_args = parse_response_yaml(response, max_retries=3)
        # todo: update self.belief
        return response

    def _choose_stocks(self) -> list:
        self.current_stocks_details = self._get_stock_details(self.stock_codes, "full")
        potential_stocks_details = self._get_stock_details(
            self.potential_stock_list, "full"
        )
        prompt = TradingPrompt.get_stock_selection_prompt(
            self.current_stocks_details,
            potential_stocks_details,
            self.belief,
            self.user_profile["fol_ind"],
        )
        self.conversation_history.append({"role": "user", "content": prompt})

        stock_agent = self.base_agent

        start_time = time.time()
        response = stock_agent.get_response(
            messages=self.conversation_history, temperature=1.3
        )
        print_debug(
            f"stock_agent.get_response耗时: {time.time() - start_time:.2f}秒",
            self.debug,
        )
        response = response.get("response")
        stock_args = parse_response_yaml(response=response, max_retries=3)

        stock_list = stock_args.get("selected_index", [])
        stock_list = [stock for stock in stock_list if stock in self.all_stock_list]
        reason = stock_args.get("reason", "")

        # print(f'\033[31m{stock_list},reason:{reason}\033[0m')

        if len(stock_list) == 0:
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": f"我今天不选择交易任何行业指数。\n理由如下 {reason}",
                }
            )

        else:
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": f"我今天选择交易的资产为: {', '.join(stock_list)}\n理由如下 {reason}",
                }
            )

        return stock_list

    def _get_stock_details(self, stock_list: list, type: str = "basic") -> str:
        df = pd.read_csv(STOCK_PROFILE_PATH2)
        stock_details_str = ""
        # 框定就是只有交易日才会调用
        if type == "full":
            df_stock = self.df_stock.copy(deep=True)

        yesterday = pd.to_datetime(self.cur_date) - pd.Timedelta(days=1)

        for stock in stock_list:
            stock_info = df[df["stock_id"] == stock]
            if not stock_info.empty:
                if type == "basic":
                    stock_details_str += f"- 指数代码：{stock}，名称：{stock_info['name'].iloc[0]}，行业：{stock_info['industry'].iloc[0]}，{self.stock_profile_dict[stock]}\n"
                elif type == "full":
                    if stock in self.user_profile["cur_positions"]:
                        market_value = self.user_profile["stock_returns"][stock][
                            "market_value"
                        ]  # 持仓市值
                        total_profit_rate = self.user_profile["stock_returns"][stock][
                            "profit"
                        ]  # 百分比持仓盈亏
                        yest_return_rate = self.user_profile["yest_returns"][
                            stock
                        ]  # 昨日涨跌幅
                        shares = self.user_profile["cur_positions"][stock][
                            "shares"
                        ]  # 持仓股数
                        ratio = self.user_profile["cur_positions"][stock][
                            "ratio"
                        ]  # 持仓占比
                        stock_details_str += f"- 指数代码：{stock},名称：{stock_info['name'].iloc[0]},行业：{stock_info['industry'].iloc[0]};持仓{shares:,}股，持仓占比为{ratio}%,持仓总市值{market_value:,}元，{self.stock_profile_dict[stock]}上个交易日这只股票{'涨了' if yest_return_rate >= 0 else '跌了'}{abs(yest_return_rate)}%，它总共让你{'赚了' if total_profit_rate >= 0 else '亏了'}{abs(total_profit_rate)}%\n"
                    else:
                        stock_data = (
                            df_stock[
                                (df_stock["stock_id"] == stock)
                                & (df_stock["date"] <= yesterday)
                            ]
                            .sort_values("date", ascending=False)
                            .iloc[0]
                        )
                        yest_return_rate = stock_data["pct_chg"]
                        price = stock_data["close_price"]
                        stock_details_str += f"- 指数代码：{stock}，名称：{stock_info['name'].iloc[0]}，行业：{stock_info['industry'].iloc[0]}，{self.stock_profile_dict[stock]}没有任何持仓信息，属于系统推荐指数，推荐原因为{'这只指数'+'涨了' if yest_return_rate >= 0 else '跌了'}{abs(yest_return_rate)}%，{'涨势良好' if yest_return_rate >= 0 else '是潜在的加仓机会'}，前一天收盘价为{price}元\n"
            else:

                stock_details_str += f"指数代码：{stock}未查询到任何相关信息。"
        return stock_details_str.strip()

    def _get_user_indicators(self):
        type = self.user_strategy
        n = random.randint(2, len(MAPPING_INDICATORS2[type]))
        selected_type_indicators = random.sample(MAPPING_INDICATORS2[type], n)

        result_list = list(selected_type_indicators)
        return result_list

    def _data_collection(self, debug) -> dict:
        self.conversation_history.append(
            {
                "role": "user",
                "content": f"""{self._generate_initial_prompt(pd.to_datetime(self.cur_date))}""",
            }
        )

        data_args = {}
        data_args["indicators"] = self._get_user_indicators()
        end_date = pd.to_datetime(self.cur_date) - pd.Timedelta(days=1)
        days_before = random.randint(5, 15)
        start_date = end_date - pd.Timedelta(days=days_before)

        data = self.get_stock_data(
            stock_codes=self.stocks_to_deal,
            indicators=data_args["indicators"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        data_2 = data
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": f"""我的需求如下：
    - 查询指标：{', '.join(data_args['indicators'])}
    - 查询时间范围：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}
    """,
            }
        )
        ts_agent = self.base_agent
        # fix
        ts_response = ts_agent.get_response(
            user_input=f"""以下是关于{len(self.stocks_to_deal)}个指数的时序数据，请从投资者的角度，分析每只指数的表现，并输出一段简短的文字总结（比如趋势、资金流动、关键时间点等），请保持客观理性，不要输出任何主观判断。\n{self._format_data_for_prompt(data_2)}""",
            temperature=0.2,
        )
        ts_response = ts_response.get("response")

        # self.conversation_history.append({
        #     "role": "user",
        #     "content": f"""根据你的需求，我帮你查询到了如下股票相关信息：\n{self._format_data_for_prompt(data_2)}\n
        #     """
        # })
        self.conversation_history.append(
            {
                "role": "user",
                "content": f"根据你的需求，我帮你查询并总结了如下行业指数相关信息：\n{ts_response}",
            }
        )

        return

    def _make_final_decision(self, debug: bool = False) -> dict:
        """
        制定最终的交易决策

        该方法是交易决策制定的核心，基于之前收集的所有信息（技术指标、新闻、
        论坛互动等）进行综合分析，并生成具体的交易决策。

        处理流程：
        1. 基于对话历史进行投资分析
        2. 获取价格限制和仓位信息
        3. 生成交易决策（买入/卖出/持有）
        4. 验证和优化决策结果
        5. 如果决策失败，返回默认持有决策

        Args:
            debug (bool): 是否开启调试模式，默认False

        Returns:
            dict: 交易决策结果字典，格式为：
                {
                    "stock_decisions": {
                        stock_code: {
                            "action": "buy/sell/hold",
                            "target_position": float,
                            "cur_position": float,
                            "target_price": float
                        }
                    },
                    "reason": "决策理由说明"
                }

        Note:
            - 使用AI代理进行分析和决策生成
            - 包含多轮重试机制以确保决策质量
            - 会自动验证决策的合理性和可执行性
        """
        print_debug("Generating final decision...", debug)

        # 辅助agent
        decision_agent = self.base_agent
        his_without_ts = copy.deepcopy(self.conversation_history[:-2])
        conversation_history = copy.deepcopy(self.conversation_history)

        # 生成一个user的对话
        self.conversation_history.append(
            {
                "role": "user",
                "content": f"""现在是做出最终交易决策的时候。请基于之前的分析，结合你的投资风格和人设，首先进行分析，然后对每行业指数做出具体的交易决策并给出你的理由。""",
            }
        )

        # 分析
        analysis_prompt = TradingPrompt.get_analysis_prompt(self.stocks_to_deal)

        conversation_history.append(
            {"role": "user", "content": f"""{analysis_prompt}"""}
        )
        his_without_ts.append({"role": "user", "content": f"""{analysis_prompt}"""})

        start_time = time.time()
        analysis_result = decision_agent.get_response(messages=conversation_history)
        print_debug(f"Analysis response time: {time.time() - start_time:.2f}秒", debug)

        analysis_result = analysis_result.get("response")
        # analysis_args = parse_response(response=response, max_retries=3)
        # analysis_result = TradingPrompt.json_to_prompt(analysis_args)
        his_without_ts.append({"role": "assistant", "content": analysis_result})

        conversation_history = his_without_ts.copy()

        # 决策--预备知识
        self.price_info = self._get_price_limits(self.stocks_to_deal)
        cur_positions = self.user_profile.get("cur_positions", {})
        position_info = {
            stock_code: {
                "current_position": cur_positions.get(stock_code, {}).get("ratio", 0)
            }
            for stock_code in self.stocks_to_deal
        }
        # 排除掉要交易的股票
        total_position = (
            sum(
                details["ratio"]
                for stock_code, details in cur_positions.items()
                if stock_code not in self.stocks_to_deal
            )
            if cur_positions
            else 0.0
        )
        available_position = 100 - total_position

        # 决策--生成决策
        decision_prompt, yaml_template = TradingPrompt.get_decision_prompt(
            self.stocks_to_deal, self.price_info, position_info, available_position
        )
        conversation_history.append({"role": "user", "content": decision_prompt})

        max_retries = 1
        error_message = ""
        for attempt in range(max_retries):
            try:
                # print_debug(f"Attempt {attempt + 1} to get decision...", debug)

                conversation_history[-1][
                    "content"
                ] = f"""{decision_prompt}\n{error_message}"""

                # if attempt > 1:
                #     decision_agent = BaseAgent(config_path='./config_random/gemini-2.0-flash-exp.yaml')

                start_time = time.time()
                response2 = decision_agent.get_response(messages=conversation_history)
                print_debug(
                    f"Decision response time: {time.time() - start_time:.2f}秒", debug
                )

                response2 = response2.get("response")
                decision_args = {}
                help_prompt = f"""Make sure your YAML output should following this format:\n {yaml_template}"""
                decision_args["stock_decisions"] = parse_response_yaml(
                    response2, max_retries=3, prompt=help_prompt
                )
                decision_args["stock_decisions"] = {
                    key.upper(): value
                    for key, value in decision_args["stock_decisions"].items()
                }

                decision_args["stock_decisions"] = convert_values_to_float(
                    decision_args["stock_decisions"]
                )
                print_debug(f"Decision response: {decision_args}", debug)

                decision_args = self._polish_decision(
                    decision_args, cur_positions, available_position
                )
                decision_result = TradingPrompt.decision_json_to_prompt(
                    decision_args, self.potential_stock_list
                )
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": f"""{analysis_result}\n{decision_result}""",
                    }
                )

                return decision_args  # 如果验证通过，返回决策

            except ValueError as e:
                # 记录验证失败的原因
                error_message = str(e)
                print(f"验证失败: {error_message}")
                # conversation_history[-1]['content'] += f"\n{error_message}"

        # 如果所有尝试都失败，返回默认持有决策
        default_decision = {
            "stock_decisions": {
                stock_code: {
                    "action": "hold",
                    "cur_position": cur_positions.get(stock_code, {}).get("ratio", 0),
                    "target_position": cur_positions.get(stock_code, {}).get(
                        "ratio", 0
                    ),
                    "target_price": 0,
                }
                for stock_code in self.stocks_to_deal
            },
            "reason": "由于多次尝试决策失败，决定暂时保持现有持仓不变。",
        }
        decision_result = TradingPrompt.decision_json_to_prompt(
            default_decision, self.potential_stock_list
        )
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": f"""{analysis_result}\n{decision_result}""",
            }
        )

        return default_decision

    def _read_news(self):
        """
        处理重要新闻广播并进行AI分析

        该方法专门为顶级用户处理重要的市场新闻广播，通过AI代理
        对新闻内容进行分析并生成投资见解。

        处理流程：
        1. 验证新闻数据的有效性
        2. 清理和去重新闻内容
        3. 使用AI代理分析新闻影响
        4. 将分析结果添加到对话历史

        Note:
            - 只有顶级用户才会调用此方法
            - 会过滤空值和重复新闻
            - 分析结果会影响后续的交易决策
            - 如果处理失败会添加错误提示到对话历史
        """
        try:
            if not self.import_news or not isinstance(self.import_news, list):
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": "进行信息检索后，我没有找到任何重要到需要群体广播的新闻。",
                    }
                )
                return

            # 确保所有新闻都是字符串格式并去除空值和NaN
            news_list = [
                str(news) for news in self.import_news if news and not pd.isna(news)
            ]

            # 如果过滤后没有新闻
            if not news_list:
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": "进行信息检索后，我没有找到任何有效的新闻。",
                    }
                )
                return

            # 去重
            news_list = list(dict.fromkeys(news_list))

            # 获取新闻分析提示
            news_prompt = TradingPrompt.get_news_analysis_prompt(news_list)

            # 添加用户提示到对话历史
            self.conversation_history.append({"role": "user", "content": news_prompt})

            # 使用BaseAgent进行异步调用
            news_agent = self.base_agent
            start_time = time.time()

            result = news_agent.get_response(messages=self.conversation_history)
            print_debug(f"新闻分析耗时: {time.time() - start_time:.2f}秒", self.debug)

            self.news_sumary = result.get("response")
            self.conversation_history.append(
                {"role": "assistant", "content": self.news_sumary}
            )
            # print(result)
        except Exception as e:
            print(f"处理新闻时发生错误: {str(e)}")
            # 可以选择添加错误信息到对话历史
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": "在处理新闻时遇到了技术问题，暂时无法分析最新新闻。",
                }
            )

    def _polish_decision(
        self, decision_args: dict, cur_positions: dict, available_position: float
    ) -> dict:
        """
        优化和校验交易决策结果

        该方法对AI生成的原始交易决策进行全面的校验和优化，确保决策的合理性和可执行性。
        主要处理以下几个方面的问题：

        处理规则：
        1. 持有决策标准化：action为hold或trading_position为0时，统一处理为hold
        2. 负数修正：修正trading_position的负数值
        3. 卖出限制：确保卖出数量不超过当前持仓
        4. 买入限制：确保买入数量不超过可用资金
        5. 潜在股票保护：潜在股票列表中的股票不允许卖出
        6. 价格区间限制：确保交易价格在涨跌停范围内
        7. 总仓位平衡：最终调整确保总仓位不超限

        Args:
            decision_args (dict): 原始交易决策结果
            cur_positions (dict): 当前持仓信息
            available_position (float): 可用仓位百分比

        Returns:
            dict: 优化后的交易决策结果，包含标准化的仓位和价格信息

        Note:
            这是一个复杂的决策校验流程，确保所有交易决策都符合市场规则和风险控制要求。
        """

        if not isinstance(decision_args, dict):
            decision_args = {"stock_decisions": {}}

        stock_decisions = decision_args.get("stock_decisions", {})
        stock_decisions = {
            stock_code: decision
            for stock_code, decision in stock_decisions.items()
            if stock_code in self.stocks_to_deal
        }

        # 记录原始可用仓位
        original_available = available_position

        # 第一轮处理
        for stock_code, decision in stock_decisions.items():
            new_decision = {}

            # 基础检查和转换
            new_decision["action"] = (
                decision.get("action")
                if isinstance(decision, dict)
                and decision.get("action") in ["buy", "sell", "hold"]
                else "hold"
            )

            trading_position = (
                float(decision.get("trading_position", 0.0))
                if isinstance(decision, dict)
                and isinstance(decision.get("trading_position"), (int, float))
                else 0.0
            )

            cur_position = cur_positions.get(stock_code, {}).get("ratio", 0)

            # (1) 处理负数 trading_position
            if trading_position < 0:
                if new_decision["action"] == "sell":
                    trading_position = abs(trading_position)
                else:
                    new_decision["action"] = "hold"

            # (2) 处理潜在股票列表
            if (
                stock_code in self.potential_stock_list
                and new_decision["action"] == "sell"
            ):
                new_decision["action"] = "hold"

            # (3) 处理 hold 情况
            if new_decision["action"] == "hold" or trading_position == 0:
                new_decision["action"] = "hold"
                new_decision["target_position"] = cur_position
                new_decision["cur_position"] = cur_position
                new_decision["target_price"] = self.price_info.get(stock_code, {}).get(
                    "pre_close", 0
                )
                stock_decisions[stock_code] = new_decision
                continue

            # (4) 处理卖出情况
            if new_decision["action"] == "sell":
                if trading_position > cur_position:
                    trading_position = cur_position
                new_decision["target_position"] = round(
                    cur_position - trading_position, 2
                )
                new_decision["cur_position"] = cur_position
                available_position += trading_position

            # (5) 处理买入情况
            if new_decision["action"] == "buy":
                if trading_position > original_available:
                    trading_position = original_available

            # (6) 处理价格区间
            if stock_code in self.price_info:
                limit_up = self.price_info[stock_code].get("limit_up", float("inf"))
                limit_down = self.price_info[stock_code].get("limit_down", 0)
                pre_close = self.price_info[stock_code].get("pre_close", 0)

                if new_decision["action"] != "hold":
                    # 生成以昨收为均值,标准差为3%的正态分布随机价格
                    std = pre_close * 0.03
                    random_price = random.normalvariate(pre_close, std)
                    random_price = round(random_price, 2)
                    # 确保价格在涨跌停区间内
                    new_decision["target_price"] = round(
                        min(max(random_price, limit_down), limit_up), 2
                    )
                else:
                    new_decision["target_price"] = pre_close

            new_decision["trading_position"] = trading_position
            stock_decisions[stock_code] = new_decision

        # (7) 最终调整所有仓位
        total_buy_position = sum(
            decision["trading_position"]
            for decision in stock_decisions.values()
            if decision["action"] == "buy"
        )

        # 对所有交易更新target_position和cur_position
        for stock_code, decision in stock_decisions.items():
            cur_pos = cur_positions.get(stock_code, {}).get("ratio", 0)

            if decision["action"] == "buy":
                if total_buy_position > available_position:
                    # 如果总买入量超过可用仓位，按比例调整
                    ratio = available_position / total_buy_position
                    adjusted_trading = (
                        math.floor(decision["trading_position"] * ratio * 100) / 100
                    )
                else:
                    # 如果没超过，使用原始trading_position
                    adjusted_trading = decision["trading_position"]

                target_position = cur_pos + adjusted_trading
                decision["target_position"] = int(target_position * 100) / 100
                decision["cur_position"] = cur_pos
                del decision["trading_position"]

            elif decision["action"] == "sell":
                del decision["trading_position"]

        decision_args["stock_decisions"] = stock_decisions
        return decision_args

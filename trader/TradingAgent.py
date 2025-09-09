import time
from codecs import BOM
from attr import define
from requests_toolbelt import user_agent
import pandas as pd
import math
import random
import pandas as pd
import json
from openai import OpenAI
from datetime import datetime, timedelta
from typing import Dict, Literal
import os
import yaml
import asyncio
from util.InformationDB import InformationDB
import re
from .utility import *
from .Prompt import TradingPrompt
from .StockRec import StockRecommender
from .IndustryDict import *
from util.UserDB import *
from util.ForumDB import *
import copy

# INFORMATION_DB = InformationDB(model_name="models/bge-m3",
#                                database_dir="data/InformationDB_news_2023")
INFORMATION_DB = InformationDB(config_path="config/embedding.yaml",
                               database_dir="data/InformationDB_news_2023")
INFORMATION_DB.load_database()
STOCK_REC = StockRecommender()
STOCK_REC._load_or_build_stock_relations()
STOCK_DB_NAME = 'StockData'


class PersonalizedStockTrader:
    def __init__(self,
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
                 ):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.is_activate_user = is_activate_user
        
        self.user_profile = user_profile
        self.user_strategy = user_strategy
        self.user_graph = user_graph
        self.InformationDataBase = INFORMATION_DB
        self.forum_db_path = forum_db_path
        self.user_db_path = user_db_path

        self.potential_stock_list = []
        self.all_stock_list = []
        self.stocks_to_deal = []
        self.df_stock = df_stock
        self.system_context = TradingPrompt.get_system_prompt(user_profile, self.user_strategy)

        self.import_news = import_news
        self.is_trading_day = is_trading_day
        self.is_top_user = is_top_user

        self.user_id = self.user_profile['user_id']
        self.forum_args = None
        self.decision_result = None

        self.is_random_trader = is_random_trader
        self.config_path = config_path
        self.base_agent = BaseAgent(config_path=self.config_path)

        self.conversation_history = [
            self.system_context,
            {"role": "user", "content": f"接下来我将提供给你更加具体的人设：{user_profile['prompt']}"},
            {"role": "assistant", "content": f"明白了，我会严格根据以上特征和投资组合状况来进行对话和决策。"}
        ]

    def _process_decision_result(self, decision_result: dict) -> dict:
        for stock_code, decision in decision_result.get("stock_decisions", {}).items():
            action = decision.get("action")
            if action != "hold":
                target_position = decision.get("target_position", 0)
                cur_position = decision.get("cur_position", 0)
                trade_position = abs(target_position - cur_position)
                target_price = decision.get("target_price", 0)

                # 计算交易数量
                if target_price > 0:
                    quantity = (trade_position / 100) * self.user_profile['total_value'] / target_price
                    if quantity < 100:
                        quantity = 100
                    else:
                        quantity = (quantity // 100) * 100  # 向下取整为100的倍数

                    # 如果是卖出操作，确保不超过当前持仓数量
                    if action == "sell" and stock_code in self.user_profile.get('cur_positions', {}):
                        current_shares = self.user_profile['cur_positions'][stock_code].get('shares', 0)
                        quantity = min(quantity, current_shares)
                        # 确保卖出数量是100的倍数
                        quantity = (quantity // 100) * 100

                    decision["quantity"] = int(quantity)  # 更新决策结果中的数量
            else:
                decision["quantity"] = 0
        return decision_result

    def should_trade_today(self) -> bool:
        return random.random() < self.user_profile['trad_pro']

    def get_stock_data(self, stock_codes: list, indicators: list, start_date: str = None, end_date: str = None) -> dict:
        try:
            # 首先处理日期参数

            result = {}

            # 1. 从company_info.csv获取公司基本信息
            company_info_df = pd.read_csv(COMPANY_INFO_PATH)

            # 2. 从数据库获取
            df = self.df_stock.copy(deep=True)

            for stock_code in stock_codes:
                # 处理股票代码格式
                if not stock_code.startswith(('SH')):
                    stock_code = f"SH{stock_code}"

                stock_result = {}

                # 获取公司基本信息
                company_row = company_info_df[company_info_df['ts_code'] == stock_code]
                if not company_row.empty:
                    company_indicators = [ind for ind in indicators if ind in company_row.columns]
                    for indicator in company_indicators:
                        stock_result[indicator] = company_row[indicator].values[0]

                # 获取交易数据
                stock_data = df[df['stock_id'] == stock_code]
                trading_indicators = [ind for ind in indicators if ind in df.columns]

                if not stock_data.empty and trading_indicators:
                    # 将日期转换为datetime对象
                    current_date = pd.to_datetime(self.cur_date)
                    start = pd.to_datetime(start_date)
                    end = pd.to_datetime(end_date)

                    # 确保end_date不超过current_date前一天
                    end = min(end, current_date - pd.Timedelta(days=1))

                    period_data = stock_data[
                        (stock_data['date'] >= start) &
                        (stock_data['date'] <= end)
                    ]

                    if not period_data.empty:
                        all_indicators = trading_indicators + ['date']
                        period_data = period_data.assign(date=period_data['date'].dt.strftime('%Y-%m-%d'))
                        trading_result = {
                            'data': period_data[all_indicators].to_dict('records'),
                            'start_date': start.strftime('%Y-%m-%d'),
                            'end_date': end.strftime('%Y-%m-%d')
                        }
                        stock_result.update(trading_result)

                result[stock_code] = stock_result

            return result

        except Exception as e:
            print(f"获取股票数据时出错: {str(e)}")
            print(f"参数信息: stock_codes={stock_codes}, indicators={indicators}, start_date={start_date}, end_date={end_date}")
            return {}

    def _get_stock_summary(self, stock_codes: list, current_date: pd.Timestamp) -> str:
        yesterday = current_date - pd.Timedelta(days=1)
        columns = ['change', 'pct_chg', 'vol', 'date', 'stock_id', 'close_price', 'pre_close']

        df = self.df_stock.copy(deep=True)

        summary = []
        for stock_code in stock_codes:
            # 处理股票代码格式
            if not stock_code.startswith(('SH')):
                stock_code = f"SH{stock_code}"

            selected_row = df.loc[(df['stock_id'] == stock_code) & (df['date'] <= yesterday)].sort_values('date', ascending=False)
            if not selected_row.empty:
                selected_row = selected_row.iloc[0][columns]
                stock_summary = TradingPrompt.get_stock_summary(stock_code, selected_row)
                summary.append(stock_summary)
            else:
                summary.append(f"## 股票代码：{stock_code} 无上个交易日交易数据。")

        return "\n".join(summary)

    def _generate_initial_prompt(self, current_date: pd.Timestamp) -> str:
        formatted_date = self._format_date(current_date)
        stock_summary = self._get_stock_summary(self.stocks_to_deal, current_date)
        positions_info = self._get_stock_details(self.stocks_to_deal, type='full')

        return TradingPrompt.get_initial_prompt(
            formatted_date=formatted_date,
            stocks_to_deal=self.stocks_to_deal,
            stock_summary=stock_summary,
            positions_info=positions_info,
            return_rate=self.user_profile['return_rate'],
            total_value=self.user_profile['total_value'],
            current_cash=self.user_profile['current_cash'],
            system_prompt=self.user_profile['sys_prompt'],
            user_strategy=self.user_strategy
        )

    def _get_environment_info(self, current_date: pd.Timestamp, debug: bool = False) -> tuple[str, bool]:
        print_debug("Getting environment information...", debug)
        news_anno = True
        if news_anno:
            new_message, queries, stock_ids = self._desire_agent(current_date)
            # self.conversation_history.append({
            #     "role": "user",
            #     "content": f"我帮你搜索到了如下信息和公告：\n\n{new_message}\n请你根据你的投资风格和人设，结合你目前的持仓谈谈你的初步看法,言简意赅一些。"})
            # agent = BaseAgent()
            # response = agent.get_response(
            #     messages=self.conversation_history
            # )
            # news_analyse = response.get("response")
            # self.conversation_history.append({
            #     "role": "assistant",
            #     "content": news_analyse})

            agent = self.base_agent
            input_message = [{"role": "user", "content": f"{self.system_context['content']}\n 我帮你搜索到了如下信息和公告：\n{new_message}\n请你根据你的投资风格和人设，结合你目前的持仓谈谈你的初步看法,言简意赅一些。"}]
            self.point1 = agent.get_response(
                messages=input_message
            ).get("response")
            self.conversation_history[-1]["content"] = f"## 我想要查询的关键词和股票代码如下：\n- 关键词：{queries}\n- 股票代码：{stock_ids} \n\n  ## 目前初步想法：\n {self.point1}"
            # self.conversation_history.append({
            #     "role": "user",
            #     "content": self.point1})
            return new_message, True

        return None, False

    def input_info(
        self,
        stock_codes: list,
        current_date: pd.Timestamp,
        debug: bool = False,
        day_1st: bool = True
    ) -> dict:
        """
        异步主逻辑：处理交易决策。
        """
        self.stock_codes = stock_codes
        self.cur_date = current_date.strftime('%Y-%m-%d')
        self.belief = None
        self.debug = debug

        # 记录总开始时间
        start_time_total = time.time()

        # 获取昨天的 belief
        if not day_1st:
            start_time = time.time()
            user_post = get_user_posts_db(
                user_id=self.user_id,
                end_date=current_date - timedelta(days=1),
                db_path=self.forum_db_path
            )
            self.belief = user_post.get("belief", None) if user_post else None
            print_debug(f"获取昨天的 belief 耗时: {time.time() - start_time:.2f}秒", debug)

        # 刷帖 TODO 更新 belief : 放在 user_profile 里面
        start_time = time.time()
        self.rec_post = []
        self.forum_args = None

        if not day_1st:
            self.rec_post = recommend_post_graph(
                target_user_id=self.user_id,
                start_date=datetime(2023, 6, 14),
                end_date=current_date - timedelta(days=1),
                db_path=self.forum_db_path,
                graph=self.user_graph,
                max_return=5
            )

            self.forum_args, self.forum_summary = self._forum_action()
            # print_debug(f'self.forum_args: {self.forum_args}', debug)
            self.conversation_history.append({"role": "user", "content": self.forum_summary})
            # todo: add to conversation history: self.forum_summary
        print_debug(f"刷帖模块耗时: {time.time() - start_time:.2f}秒", debug)
        
        # fix： 不是activate user直接返回
        if self.is_activate_user:
            
            print_debug(f"User {self.user_id} is activate: {self.is_activate_user}", debug)
            
            # 全体新闻
            start_time = time.time()
            # 读取全体重要新闻广播
            if self.is_top_user:  # todo: check
                self._read_news()
            # 全体新闻逻辑
                print_debug(f"全体新闻模块耗时: {time.time() - start_time:.2f}秒", debug)

            # 系统随机推荐股票
            if self.is_trading_day:
                start_time = time.time()
                self._get_rec_stock()
                # print_debug(f"系统随机推荐股票耗时: {time.time() - start_time:.2f}秒", debug)

            # 查找新闻（公告）
            if not self.is_random_trader:
                start_time = time.time()
                environment_info, whether_decision = self._get_environment_info(current_date, debug)
                print_debug(f"查找新闻（公告）耗时: {time.time() - start_time:.2f}秒", debug)

            # 更新 belief
            if not self.is_random_trader:
                start_time = time.time()
                tmp_belief = self._update_belief()
                self.belief = tmp_belief.get("belief", None)
                print_debug(f"更新 belief 耗时: {time.time() - start_time:.2f}秒", debug)

            # 选择待交易的股票 TODO：可能为空
            if self.is_trading_day and not self.is_random_trader:
                start_time = time.time()
                self.stocks_to_deal = self._choose_stocks()
                print_debug(f"选择待交易的股票耗时: {time.time() - start_time:.2f}秒", debug)

            # 收集查询的数据 TODO：可能为空
            if self.is_trading_day and not self.is_random_trader:
                start_time = time.time()
                self.collected_data = self._data_collection(debug)
                print_debug(f"收集查询的数据耗时: {time.time() - start_time:.2f}秒", debug)

        return self, self.user_id, self.forum_args

        # # 生成最终决策
        # if self.is_trading_day:
        #     start_time = time.time()
        #     decision_result = self._make_final_decision(self.collected_data, debug)
        #     print_debug(f"生成最终决策耗时: {time.time() - start_time:.2f}秒", debug)

        # # 处理决策结果，计算每个股票的交易数量
        # if self.is_trading_day:
        #     start_time = time.time()
        #     decision_result = self._process_decision_result(decision_result)
        #     print_debug(f"处理决策结果耗时: {time.time() - start_time:.2f}秒", debug)

        # # 与 environment 交互
        # start_time = time.time()
        # print_debug("Interacting with environment...", debug)
        # post_response_args = self._intention_agent(current_date, self.conversation_history)  # post, type, belief
        # print_debug(json.dumps(post_response_args, indent=2, ensure_ascii=False), debug)
        # create_post_db(
        #     user_id=self.user_id,
        #     content=post_response_args["post"],
        #     type=post_response_args["type"],
        #     belief=str(post_response_args["belief"]),
        #     created_at=current_date,
        #     db_path=self.forum_db_path)
        # print_debug(f"与 environment 交互耗时: {time.time() - start_time:.2f}秒", debug)

        # # 打印总耗时
        # print_debug(f"总耗时: {time.time() - start_time_total:.2f}秒", debug)

        # return decision_result, self.forum_args

    def output_decision(self):
        # 生成最终决策
        debug = self.debug
        current_date = pd.to_datetime(self.cur_date)

        if self.is_trading_day:
            start_time = time.time()
            if not self.is_random_trader:
                self.decision_result = self._make_final_decision(debug)
            else:
                self.decision_result = self._make_final_decision_random(debug)
            print_debug(f"{'是' if self.is_random_trader else '不是'}random trader；生成最终决策耗时: {time.time() - start_time:.2f}秒", debug)

        # 处理决策结果，计算每个股票的交易数量
        if self.is_trading_day:
            start_time = time.time()
            self.decision_result = self._process_decision_result(self.decision_result)
            print_debug(f"处理决策结果耗时: {time.time() - start_time:.2f}秒", debug)

        # 与 environment 交互
        start_time = time.time()
        print_debug("Interacting with environment...", debug)
        post_response_args = self._intention_agent(current_date, self.conversation_history)  # post, type, belief
        # print_debug(json.dumps(post_response_args, indent=2, ensure_ascii=False), debug)
        # create_post_db(
        #     user_id=self.user_id,
        #     content=post_response_args["post"],
        #     type=post_response_args["type"],
        #     belief=str(post_response_args["belief"]),
        #     created_at=current_date,
        #     db_path=self.forum_db_path)
        print_debug(f"与 environment 交互耗时: {time.time() - start_time:.2f}秒", debug)

        return self.user_id, self.decision_result, post_response_args

    def _make_final_decision_random(self, debug: bool = False) -> dict:

        print_debug("Generating final decision...", debug)

        # 随机决策
        self.stocks_to_deal = list(set(self.stock_codes) | set(self.potential_stock_list))

        # 决策-预备知识
        price_info = self._get_price_limits(self.stocks_to_deal)
        cur_positions = self.user_profile.get('cur_positions', {})
        position_info = {
            stock_code: {'current_position': cur_positions.get(stock_code, {}).get('ratio', 0)}
            for stock_code in self.stocks_to_deal
        }
        # 排除掉要交易的股票
        total_position = sum(
            details['ratio'] for stock_code, details in cur_positions.items()
            if stock_code not in self.stocks_to_deal
        ) if cur_positions else 0.0
        available_position = 100 - total_position

        # 生成随机的决策
        decision_args = {}
        decision_args['stock_decisions'] = self._generate_random_decision(price_info)
        decision_args['stock_decisions'] = convert_values_to_float(decision_args['stock_decisions'])
        print_debug(f"Decision response: {decision_args}", debug)
        # 验证决策
        # decision_args = self._validate_decision(decision_args, price_info, cur_positions, available_position)
        decision_args = self._polish_decision(decision_args, price_info, cur_positions, available_position)
        analysis_result = '我今天的决策由AI驱动，在后续的belief更新中，请根据我的决策结果进行更新。'
        decision_result = TradingPrompt.decision_json_to_prompt(decision_args, self.potential_stock_list)

        # 生成一个user的对话
        self.conversation_history.append({"role": "user",
                                          "content": f"""现在是做出最终交易决策的时候。请基于之前的分析，结合你的投资风格和人设，首先进行分析，然后对每支股票做出具体的交易决策并给出你的理由。"""})
        self.conversation_history.append({"role": "assistant", "content": f"""{analysis_result}\n{decision_result}"""})

        return decision_args  # 如果验证通过，返回决策

    def _generate_random_decision(self, price_info: dict) -> dict:
        # 获取前一天的股票数据
        df = self.df_stock.copy(deep=True)
        yesterday = pd.to_datetime(self.cur_date) - pd.Timedelta(days=1)

        stock_decisions = {}

        for stock_code in self.stocks_to_deal:
            # 获取前一天的涨跌数据
            stock_data = df[(df['stock_id'] == stock_code) & (df['date'] <= yesterday)].sort_values('date', ascending=False)

            if not stock_data.empty:
                pct_change = stock_data.iloc[0]['pct_chg']  # 获取涨跌幅

                price = price_info[stock_code]['pre_close']

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
                    "target_price": price
                }

        return stock_decisions

    def _get_price_limits(self, stock_codes: list) -> dict:
        current_date = pd.to_datetime(self.cur_date)
        df = self.df_stock.copy(deep=True)

        # 创建结果字典
        price_limits = {}

        # 筛选所有相关股票的数据
        df_filtered = df[df['stock_id'].isin(stock_codes) & (df['date'] < current_date)]

        # 对每只股票获取最新的收盘价
        for stock_code in stock_codes:
            selected_row = df_filtered[df_filtered['stock_id'] == stock_code].sort_values('date', ascending=False)

            if not selected_row.empty:
                pre_close_price = float(selected_row.iloc[0]['close_price'])
                # 计算涨跌停价格（假设是10%的限制）
                limit_up = round(pre_close_price * 1.1, 2)
                limit_down = round(pre_close_price * 0.9, 2)

                price_limits[stock_code] = {
                    'pre_close': pre_close_price,
                    'limit_up': limit_up,
                    'limit_down': limit_down
                }
            else:
                raise ValueError(f"无法获取股票 {stock_code} 在 {current_date} 的价格数据")

        return price_limits

    def _format_date(self, date: pd.Timestamp) -> str:
        weekday_map = {
            0: '一',
            1: '二',
            2: '三',
            3: '四',
            4: '五',
            5: '六',
            6: '日'
        }

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
            result.append(f"\n# {stock_code} 的额外股票信息")
            result.append(f"查询区间：{stock_data.get('start_date', '')} 至 {stock_data.get('end_date', '')}")

            if 'data' in stock_data:
                time_series_data = stock_data['data']

                # 获取所有非空指标名称
                valid_indicators = set()
                for record in time_series_data:
                    for k, v in record.items():
                        if k != 'date' and v is not None:
                            valid_indicators.add(k)

                # 按日期显示数据
                for record in time_series_data:
                    date = record['date']
                    result.append(f"\n## {date}")

                    for indicator in sorted(valid_indicators):
                        value = record[indicator]
                        if value is not None:  # 只显示非空值
                            mapped_ind = MAPPING_DICT.get(indicator, indicator)

                            # 根据指标类型格式化数值
                            if indicator == 'elg_amount_net':
                                value_str = f"{value:,.2f} 万元" if value else "0.00 万元"
                                trend = "净流入" if value > 0 else "净流出"
                                result.append(f"- {mapped_ind}: {value_str} ({trend})")
                            elif indicator.startswith('ma_hfq'):
                                result.append(f"- {mapped_ind}: {value:,.2f}")
                            elif indicator.startswith('macd'):
                                result.append(f"- {mapped_ind}: {value:,.3f}")
                            else:
                                result.append(f"- {mapped_ind}: {value}")

        return "\n".join(result)

    def _intention_agent(self, current_date: pd.Timestamp, conversation_history: dict):
        before_decision_history = conversation_history[:-2]
        post_agent = self.base_agent
        post_prompt = TradingPrompt.get_intention_prompt()
        self.conversation_history.append({"role": "user", "content": f'''{self.user_profile['sys_prompt']}\n{post_prompt}'''})
        post_response = post_agent.get_response(
            messages=self.conversation_history,
            temperature=1.3
            # response_format={"type": "json_object"}
        )
        post_response = post_response.get("response")
        self.conversation_history.append({"role": "assistant", "content": post_response})
        post_response_args = parse_response_yaml(response=post_response, max_retries=3,
                                                 prompt=f"""
                             You need to ensure the YAML keys are as follows:
                             post: ...
                             type: ...
                             belief: ...""")
        # print(post_response_args)
        return post_response_args

    # def _desire_agent(self, current_date: pd.Timestamp):
    #     """
    #     异步生成用户的查询问题并并发查询新闻信息。
    #     """
    #     # 第一步：生成用户的查询问题
    #     query_agent = BaseAgent()

    #     self.all_stock_list = list(set(self.stock_codes) | set(self.potential_stock_list))

    #     stock_details_str = self._get_stock_details(self.all_stock_list, type="basic")

    #     query_prompt = TradingPrompt.get_query_for_na_prompt(
    #         user_type=self.user_profile['user_type'],
    #         stock_details=stock_details_str,
    #         current_date=current_date.strftime('%Y年%m月%d日')
    #     )

    #     self.conversation_history.append({"role": "user", "content": query_prompt})

    #     # 使用 调用异步方法
    #     query_response = query_agent.get_response(
    #         messages=self.conversation_history,
    #     )
    #     response_content = query_response.get("response")  # 获取响应内容

    #     self.conversation_history[-1]['content'] = TradingPrompt.get_query_for_na_prompt2(
    #         user_type=self.user_profile['user_type'],
    #         stock_details=stock_details_str,
    #         current_date=current_date.strftime('%Y年%m月%d日')
    #     )

    #     # 提取 <output> 标签中的内容
    #     pattern = r"<output>(.*?)</output>"
    #     match = re.search(pattern, response_content, re.DOTALL)
    #     if match:
    #         query_response = match.group(1).strip()
    #     else:
    #         # 如果没有 <output> 标签，直接使用 query_response
    #         query_response = response_content.strip()

    #     self.conversation_history.append({"role": "assistant", "content": query_response})
    #     self.conversation_history.append({"role": "user", "content": TradingPrompt.get_query_desire_prompt()})

    #     # 使用 调用异步方法
    #     summary_response = query_agent.get_response(
    #         messages=self.conversation_history,
    #     )
    #     summary_response = summary_response.get("response")  # 获取响应内容

    #     # 删除刚刚的 user 提问
    #     self.conversation_history.pop()

    #     search_args = parse_response_yaml(response=summary_response, max_retries=3)
    #     queries = search_args.get("queries", None)
    #     stock_ids = search_args.get("stock_id", None)

    #     result_str = ""

    #     # 查找新闻信息  --最初版本
    #     for query in queries:

    #         # # TODO 加reranker
    #         # news_result = self.InformationDataBase.search_news(start_date=current_date-pd.Timedelta(days=7),
    #         #                                                    end_date=current_date,
    #         #                                                    query=query,
    #         #                                                    top_k=10,
    #         #                                                    type=None)

    #         # # 从新闻数据中选择两个记录
    #         # if news_result:
    #         #     samples = [a['content'] for a in news_result]
    #         #     timelines = [a['datetime'] for a in news_result]
    #         #     samples, timelines = rerank_documents(query, samples, timelines)
    #         #     result_str += f"查询<{query}> 得到的新闻信息如下:\n"
    #         #     for i in range(0, len(samples)):
    #         #         result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
    #         #     result_str += '\n'

    #         # TODO 加reranker
    #         news_result = self.InformationDataBase.search_news(start_date=current_date-pd.Timedelta(days=7),
    #                                                            end_date=current_date,
    #                                                            query=query,
    #                                                            top_k=2,
    #                                                            type=None)

    #         # 从新闻数据中选择两个记录
    #         if news_result:
    #             samples = [a['content'] for a in news_result]
    #             timelines = [a['datetime'] for a in news_result]
    #             result_str += f"查询<{query}> 得到的新闻信息如下:\n"
    #             for i in range(0, len(samples)):
    #                 result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
    #             result_str += '\n'

    #     return result_str

    # 修改的

    def _desire_agent(self, current_date: pd.Timestamp):
        """
        异步生成用户的查询问题并并发查询新闻信息。
        """
        # 第一步：生成用户的查询问题
        main_time = time.time()
        query_agent = self.base_agent

        self.all_stock_list = list(set(self.stock_codes) | set(self.potential_stock_list))

        stock_details_str = self._get_stock_details(self.all_stock_list, type="basic")

        query_prompt = TradingPrompt.get_query_for_na_prompt(
            user_type=self.user_profile['user_type'],
            stock_details=stock_details_str,
            current_date=current_date.strftime('%Y年%m月%d日')
        )

        self.conversation_history.append({"role": "user", "content": query_prompt})

        # 使用 调用异步方法
        query_response = query_agent.get_response(
            messages=self.conversation_history,
        )
        response_content = query_response.get("response")  # 获取响应内容

        self.conversation_history[-1]['content'] = TradingPrompt.get_query_for_na_prompt2(
            user_type=self.user_profile['user_type'],
            stock_details=stock_details_str,
            current_date=current_date.strftime('%Y年%m月%d日')
        )

        # 提取 <output> 标签中的内容
        pattern = r"<output>(.*?)</output>"
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            query_response = match.group(1).strip()
        else:
            # 如果没有 <output> 标签，直接使用 query_response
            query_response = response_content.strip()

        self.conversation_history.append({"role": "assistant", "content": f"## 我的回答如下：\n{query_response}"})
        self.conversation_history.append({"role": "user", "content": TradingPrompt.get_query_desire_prompt()})

        # 使用 调用异步方法
        summary_response = query_agent.get_response(
            messages=self.conversation_history,
        )
        summary_response = summary_response.get("response")  # 获取响应内容

        # 删除刚刚的 user 提问
        self.conversation_history.pop()

        search_args = parse_response_yaml(response=summary_response, max_retries=3)
        queries = search_args.get("queries", None)
        stock_ids = search_args.get("stock_id", None)

        max_num = 1
        if queries:
            queries = random.sample(queries, min(max_num, len(queries)))
        if stock_ids:
            stock_ids = random.sample(stock_ids, min(max_num, len(stock_ids)))

        # print(f"\033[31m查找新闻前获取query耗时: {time.time() - main_time:.2f}秒\033[0m")

        main_time = time.time()

        # 版本1
        # def search_and_process_news(query):
        #     news_result = self.InformationDataBase.search_news(
        #         start_date=current_date - pd.Timedelta(days=7),
        #         end_date=current_date,
        #         query=query,
        #         top_k=2,
        #         type=None
        #     )
        #     if news_result:
        #         samples = [a['content'] for a in news_result]
        #         timelines = [a['datetime'] for a in news_result]
        #         result_str = f"查询<{query}> 得到的新闻信息如下:\n"
        #         for i in range(0, len(samples)):
        #             result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
        #         return result_str
        #     else:
        #         return ""

        # if queries:
        #      # 使用 asyncio.gather 并发执行搜索任务
        #     tasks = [search_and_process_news(query) for query in queries]
        #     results = asyncio.gather(*tasks)
        #     result_str = "".join(results)
        # else:
        #     result_str=""

        # 版本2
        if not queries:
            return ''

        news_results_list = self.InformationDataBase.search_news_batch(
            start_date=current_date - pd.Timedelta(days=7),
            end_date=current_date,
            queries=queries,
            top_k=2,
            type=None
        )

        result_str = ""

        # 处理批量查询结果
        for query, news_results in zip(queries, news_results_list):
            if news_results:
                samples = [a['content'] for a in news_results]
                timelines = [a['datetime'] for a in news_results]

                # samples, timelines = rerank_documents_async(query, samples, timelines, top_n=2)

                result_str += f"查询<{query}> 得到的新闻信息如下:\n"
                for i in range(0, len(samples)):
                    result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
                result_str += '\n'

        # print(f'faiss查询结果:\n{result_str}')

        # print(f"\033[31m faiss搜索新闻耗时: {time.time() - main_time:.2f}秒，query长度为{len(queries)}\033[0m")
        return result_str, queries, stock_ids

        # result_str = ""

        # # 查找新闻信息
        # for query in queries:

        #     # # TODO 加reranker
        #     # news_result = self.InformationDataBase.search_news(start_date=current_date-pd.Timedelta(days=7),
        #     #                                                    end_date=current_date,
        #     #                                                    query=query,
        #     #                                                    top_k=10,
        #     #                                                    type=None)

        #     # # 从新闻数据中选择两个记录
        #     # if news_result:
        #     #     samples = [a['content'] for a in news_result]
        #     #     timelines = [a['datetime'] for a in news_result]
        #     #     samples, timelines = rerank_documents(query, samples, timelines)
        #     #     result_str += f"查询<{query}> 得到的新闻信息如下:\n"
        #     #     for i in range(0, len(samples)):
        #     #         result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
        #     #     result_str += '\n'

        #     # TODO 加reranker
        #     news_result = self.InformationDataBase.search_news(start_date=current_date-pd.Timedelta(days=7),
        #                                                        end_date=current_date,
        #                                                        query=query,
        #                                                        top_k=2,
        #                                                        type=None)

        #     # 从新闻数据中选择两个记录
        #     if news_result:
        #         samples = [a['content'] for a in news_result]
        #         timelines = [a['datetime'] for a in news_result]
        #         result_str += f"查询<{query}> 得到的新闻信息如下:\n"
        #         for i in range(0, len(samples)):
        #             result_str += f"- 第{i+1}条结果:{timelines[i]}: {samples[i]}\n"
        #         result_str += '\n'

        # return result_str

    def _forum_action(self) -> tuple[list[dict], str]:
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
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute('''
                            SELECT content FROM posts WHERE id = ?
                        ''', (reference_id,))
                        reference_post = cursor.fetchone()
                        if reference_post:
                            reference_content = reference_post["content"]

            # 获取针对当前帖子的决策 prompt
            post_decision_prompt = f"""
            {self.user_profile['sys_prompt']}
            以下是当前帖子的信息：
            帖子ID: {post_id}
            内容: {post_content}
            """

            # 如果帖子是 repost 类型，添加引用内容
            if post_type == "repost" and reference_content:
                post_decision_prompt += f"""
                该帖子引用了以下内容：
                引用内容: {reference_content}
                """

            post_decision_prompt += f"""
            请根据以上信息决定是否对该帖子执行操作。
            你可以选择以下操作之一：
            - repost: 转发: 你认为这个帖子值得分享给更多人，可以添加你的评论
            - unlike: 取消点赞: 你认为这个帖子不值得点赞
            - like: 点赞: 你认为这是一个有价值的帖子

            请按照以下 yaml 格式输出你的决策：
            ```yaml
            action: <操作类型>
            post_id: <帖子ID>
            reason: <操作的理由>
            ```
            """
            forum_message.append({"role": "user", "content": post_decision_prompt})

            # 获取论坛行为的响应
            forum_agent = self.base_agent
            response = forum_agent.get_response(messages=forum_message,
                                                temperature=1.3)
            response = response.get("response")

            # 解析响应为决策参数
            post_decision_args = parse_response_yaml(response, max_retries=3)
            if isinstance(post_decision_args, dict):
                post_decision_args = [post_decision_args]

            # 处理转发操作
            for arg in post_decision_args:
                if arg["action"] == "repost":
                    # 获取目标帖子的内容
                    target_post_id = arg["post_id"]
                    target_post = next((post for post in self.rec_post if post["id"] == target_post_id), None)
                    if target_post:
                        # 构建生成转发内容的 prompt
                        content_prompt = f"""
                        你决定转发以下帖子：
                        帖子ID: {target_post_id}
                        内容: {target_post["content"]}

                        请生成一段转发内容，简要说明你转发的原因或评论。
                        你应该按照 yaml 格式输出:
                        ```yaml
                        content: <你的转发内容>
                        ```
                        """
                        forum_message.append({"role": "assistant", "content": content_prompt})
                        content_response = forum_agent.get_response(messages=forum_message,
                                                                    temperature=1.3)
                        content_response = content_response.get("response")
                        content_data = parse_response_yaml(response=content_response, max_retries=3)
                        if isinstance(content_data, dict) and "content" in content_data:
                            arg["content"] = content_data["content"]

            # 将当前帖子的决策参数添加到总决策参数列表中
            decision_args.extend(post_decision_args)

            # 清理上一条帖子的内容，避免影响下一条帖子的决策
            if len(forum_message) > len(self.conversation_history):
                forum_message.pop()  # 移除上一条帖子的决策内容

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
                    action_summary += f"- 你转发了帖子 {post_id}，转发内容为：{content}\n  理由：{reason}\n"
                elif action_type == "like":
                    action_summary += f"- 你点赞了帖子 {post_id}\n  理由：{reason}\n"
                elif action_type == "unlike":
                    action_summary += f"- 你取消点赞了帖子 {post_id}\n  理由：{reason}\n"

        return decision_args, action_summary

    def _get_rec_stock(self):
        rec_stock = STOCK_REC.recommend_portfolio(input_portfolio=self.stock_codes, top_n=3)
        self.potential_stock_list = rec_stock

    def _update_belief(self) -> dict:
        pre_conversation_history = self.conversation_history.copy()
        pre_conversation_history.append({"role": "assistant", "content": TradingPrompt.get_update_belief_prompt()})
        update_agent = self.base_agent
        response = update_agent.get_response(
            messages=pre_conversation_history,
            # temperature=1.3
        )
        response = response.get("response")
        belief_args = parse_response_yaml(response, max_retries=3)
        # todo: update self.belief
        return belief_args

    def _choose_stocks(self) -> list:
        self.current_stocks_details = self._get_stock_details(self.stock_codes, 'full')
        potential_stocks_details = self._get_stock_details(self.potential_stock_list, 'basic')
        prompt = TradingPrompt.get_stock_selection_prompt(
            self.current_stocks_details,
            potential_stocks_details,
            self.belief
        )
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })

        stock_agent = self.base_agent

        start_time = time.time()
        response = stock_agent.get_response(messages=self.conversation_history,
                                            temperature=1.3)
        print_debug(f"stock_agent.get_response耗时: {time.time() - start_time:.2f}秒", self.debug)
        response = response.get("response")
        stock_args = parse_response_yaml(response=response, max_retries=3)

        stock_list = stock_args.get("selected_stocks", [])
        stock_list = [stock for stock in stock_list if stock in self.all_stock_list]
        reason = stock_args.get("reason", "")

        # print(f'\033[31m{stock_list},reason:{reason}\033[0m')

        if len(stock_list) == 0:
            self.conversation_history.append({
                "role": "assistant",
                "content": f"我今天不选择交易任何股票。\n理由如下 {reason}"
            })

        else:
            self.conversation_history.append({
                "role": "assistant",
                "content": f"我今天选择交易的股票为: {', '.join(stock_list)}\n理由如下 {reason}"
            })

        return stock_list

    def _get_stock_details(self, stock_list: list, type: str = "basic") -> str:
        df = pd.read_csv(STOCK_PROFILE_PATH)
        stock_details_str = ""

        for stock in stock_list:
            # 确保股票代码格式正确
            if not stock.startswith('SH'):
                stock = f'SH{stock}'

            stock_info = df[df['stock_id'] == stock]
            if not stock_info.empty:
                if type == "basic":
                    stock_details_str += f"- 股票代码：{stock}，名称：{stock_info['name'].iloc[0]}，行业：{stock_info['industry'].iloc[0]}；\n"
                elif type == "full":
                    if stock in self.user_profile['cur_positions']:
                        market_value = self.user_profile["stock_returns"][stock]["market_value"]  # 持仓市值
                        total_profit_rate = self.user_profile["stock_returns"][stock]['profit']  # 百分比持仓盈亏
                        yest_return_rate = self.user_profile["yest_returns"][stock]  # 昨日涨跌幅
                        shares = self.user_profile["cur_positions"][stock]['shares']  # 持仓股数
                        ratio = self.user_profile["cur_positions"][stock]['ratio']  # 持仓占比
                        stock_details_str += f"- 股票代码：{stock},名称：{stock_info['name'].iloc[0]},行业：{stock_info['industry'].iloc[0]};持仓{shares:,}股，持仓占比为{ratio}%,持仓总市值{market_value:,}元，上个交易日这只股票{'涨了' if yest_return_rate >= 0 else '跌了'}{abs(yest_return_rate)}%，它总共让你{'赚了' if total_profit_rate >= 0 else '亏了'}{abs(total_profit_rate)}%；\n"
                    else:
                        stock_details_str += f"- 股票代码：{stock}，名称：{stock_info['name'].iloc[0]}，行业：{stock_info['industry'].iloc[0]}，没有任何持仓信息，属于系统推荐股票；\n"
            else:
                stock_details_str += f'股票代码：{stock}未查询到任何相关信息。'
        return stock_details_str.strip()

    def _get_user_indicators(self):
        type=self.user_strategy
        n = random.randint(2, len(MAPPING_INDICATORS[type]))
        selected_type_indicators = random.sample(MAPPING_INDICATORS[type], n)

        m = random.randint(0, min(2, len(MAPPING_INDICATORS['宏观指标'])))
        selected_macro_indicators = random.sample(MAPPING_INDICATORS['宏观指标'], m)

        result_list = list(selected_type_indicators + selected_macro_indicators)
        return result_list
        

    def _data_collection(self, debug) -> dict:
        self.conversation_history.append({"role": "user", "content": f"""{self._generate_initial_prompt(pd.to_datetime(self.cur_date))}"""})
        
        data_args= {}
        data_args['indicators'] = self._get_user_indicators()
        end_date = pd.to_datetime(self.cur_date) - pd.Timedelta(days=1)
        days_before = random.randint(5, 15)
        start_date = end_date - pd.Timedelta(days=days_before)
        
        # tmp_conversation_history = self.conversation_history.copy()
        # data_agent = self.base_agent
        # data_response = data_agent.get_response(messages=tmp_conversation_history)
        # data_response = data_response.get("response")

        # # 异常处理
        # data_args = {}
        # data_args = parse_response_yaml(response=data_response, max_retries=3, prompt='Make sure your date is as this format: %Y-%m-%d')
        # # print_debug(f"Data collection response: {data_args}", debug)

        # # 异常处理
        # if data_args.get('indicators', []) != []:
        #     data_args['indicators'] = [a for a in data_args['indicators'] if a in INDICATORS]
        # else:
        #     num_indicators = random.randint(1, 5)
        #     data_args['indicators'] = random.sample(INDICATORS, num_indicators)

        # # 异常处理
        # max_start_date = pd.to_datetime(self.cur_date) - pd.Timedelta(days=15)
        # max_end_date = pd.to_datetime(self.cur_date) - pd.Timedelta(days=1)

        # # 处理 start_date
        # start_date_str = data_args.get('start_date', max_start_date.strftime('%Y-%m-%d'))
        # try:
        #     start_date = pd.to_datetime(start_date_str)
        # except (ValueError, TypeError):
        #     start_date = max_start_date
        #     print(f"无法解析start_date: {start_date_str}，使用默认值: {start_date}")

        # # 确保start_date不早于15天前
        # if start_date < max_start_date:
        #     start_date = max_start_date

        # # 处理 end_date
        # end_date_str = data_args.get('end_date', max_end_date.strftime('%Y-%m-%d'))
        # try:
        #     end_date = pd.to_datetime(end_date_str)
        # except (ValueError, TypeError):
        #     end_date = max_end_date
        #     print(f"无法解析end_date: {end_date_str}，使用默认值: {end_date}")

        # # 确保end_date不超过昨天
        # if end_date > max_end_date:
        #     end_date = max_end_date

        # reason = data_args.get('reason', '')

        data = self.get_stock_data(stock_codes=self.stocks_to_deal,
                                   indicators=data_args["indicators"],
                                   start_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=end_date.strftime('%Y-%m-%d'))
        # print_debug(f"Collected data: {json.dumps(data, indent=2, ensure_ascii=False)}", debug)
        data_2 = json.loads(json.dumps(data, ensure_ascii=False))
    #     self.conversation_history.append({
    #         "role": "assistant",
    #         "content": f"""我的需求如下：
    # - 查询指标：{', '.join(data_args['indicators'])}
    # - 查询时间范围：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}
    # - 理由：{reason}
    # """
    #     })

        self.conversation_history.append({
            "role": "assistant",
            "content": f"""我的需求如下：
    - 查询指标：{', '.join(data_args['indicators'])}
    - 查询时间范围：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}
    """
        })
        ts_agent = self.base_agent
        ts_response = ts_agent.get_response(user_input=f"""请全面总结这段时间序列信息：\n{self._format_data_for_prompt(data_2)}""")
        ts_response = ts_response.get("response")

        # self.conversation_history.append({
        #     "role": "user",
        #     "content": f"""根据你的需求，我帮你查询到了如下股票相关信息：\n{self._format_data_for_prompt(data_2)}\n
        #     """
        # })
        self.conversation_history.append({"role": "user", "content": f"根据你的需求，我帮你查询到了如下股票相关信息：\n{ts_response}"})

        return data_2

    def _make_final_decision(self, debug: bool = False) -> dict:
        print_debug("Generating final decision...", debug)

        # 辅助agent
        decision_agent = self.base_agent
        his_without_ts = copy.deepcopy(self.conversation_history[:-2])
        conversation_history = copy.deepcopy(self.conversation_history)

        # 生成一个user的对话
        self.conversation_history.append({"role": "user",
                                          "content": f"""现在是做出最终交易决策的时候。请基于之前的分析，结合你的投资风格和人设，首先进行分析，然后对每支股票做出具体的交易决策并给出你的理由。"""})

        # 分析
        analysis_prompt = TradingPrompt.get_analysis_prompt(self.stocks_to_deal)

        conversation_history.append({"role": "user", "content": f"""{analysis_prompt}"""})
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
        price_info = self._get_price_limits(self.stocks_to_deal)
        cur_positions = self.user_profile.get('cur_positions', {})
        position_info = {
            stock_code: {'current_position': cur_positions.get(stock_code, {}).get('ratio', 0)}
            for stock_code in self.stocks_to_deal
        }
        # 排除掉要交易的股票
        total_position = sum(
            details['ratio'] for stock_code, details in cur_positions.items()
            if stock_code not in self.stocks_to_deal
        ) if cur_positions else 0.0
        available_position = 100 - total_position

        # 决策--生成决策
        decision_prompt, yaml_template = TradingPrompt.get_decision_prompt(self.stocks_to_deal, price_info, position_info, available_position)
        conversation_history.append({"role": "user", "content": decision_prompt})

        max_retries = 3
        error_message = ''
        for attempt in range(max_retries):
            try:
                print_debug(f"Attempt {attempt + 1} to get decision...", debug)

                conversation_history[-1]['content'] = f"""{decision_prompt}\n{error_message}"""
                
                if attempt > 1:
                    decision_agent = BaseAgent(config_path='./config_random/gaochao_4o.yaml')

                start_time = time.time()
                response2 = decision_agent.get_response(messages=conversation_history)
                print_debug(f"Decision response time: {time.time() - start_time:.2f}秒", debug)

                response2 = response2.get("response")
                decision_args = {}
                # help_prompt = 'The stocks to trade are: ' + ','.join(self.stocks_to_deal)+' , please make sure the stock code is correct\n'
                help_prompt = f'''Make sure your YAML output should following this format:\n {yaml_template}'''
                decision_args['stock_decisions'] = parse_response_yaml(response2, max_retries=3, prompt=help_prompt)
                decision_args['stock_decisions'] = {key.upper(): value for key, value in decision_args['stock_decisions'].items()}
                # decision_args['stock_decisions'] = preprocess_stock_decisions(decision_args['stock_decisions'])
                decision_args['stock_decisions'] = convert_values_to_float(decision_args['stock_decisions'])
                print_debug(f"Decision response: {decision_args}", debug)
                # 验证决策
                # decision_args = self._validate_decision(decision_args, price_info, cur_positions, available_position)
                decision_args = self._polish_decision(decision_args, price_info, cur_positions, available_position)
                decision_result = TradingPrompt.decision_json_to_prompt(decision_args, self.potential_stock_list)
                self.conversation_history.append({"role": "assistant", "content": f"""{analysis_result}\n{decision_result}"""})

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
                    "cur_position": cur_positions.get(stock_code, {}).get('ratio', 0),
                    "target_position": cur_positions.get(stock_code, {}).get('ratio', 0),
                    "target_price": 0
                } for stock_code in self.stocks_to_deal
            },
            "reason": "由于多次尝试决策失败，决定暂时保持现有持仓不变。"
        }
        decision_result = TradingPrompt.decision_json_to_prompt(default_decision, self.potential_stock_list)
        self.conversation_history.append({"role": "assistant", "content": f"""{analysis_result}\n{decision_result}"""})

        return default_decision

    def _validate_decision(self, decision_args: dict, price_info: dict, cur_positions: dict, available_position: float) -> dict:
        validation_errors = set()  # 使用集合来去重

        # 1. 验证基本字段
        required_fields = ["stock_decisions"]
        missing_fields = [field for field in required_fields if field not in decision_args]
        if missing_fields:
            validation_errors.add("必须满足所有给定字段")

        # 2. 验证 stock_decisions 格式
        stock_decisions = decision_args.get("stock_decisions")
        if not isinstance(stock_decisions, dict):
            validation_errors.add("stock_decisions 必须是一个字典，键为股票代码，值为决策信息")
        else:
            total_target_position = 0

            for stock_code, decision in stock_decisions.items():
                # 检查 stock_code 是否在 self.stocks_to_deal 中
                if stock_code not in self.stocks_to_deal:
                    validation_errors.add(f"股票代码必须在待处理的股票列表：{self.stocks_to_deal} 中")

                # 3. 将决策信息统一转换为字典
                decision_dict = {}
                if isinstance(decision, list):
                    # 处理列表格式的决策信息
                    for item in decision:
                        if isinstance(item, dict):
                            decision_dict.update(item)
                elif isinstance(decision, dict):
                    # 处理字典格式的决策信息
                    decision_dict = decision
                else:
                    validation_errors.add(f"决策信息必须是列表或字典")
                    continue  # 跳过后续验证

                # 4. 验证 action 值
                action = decision_dict.get("action")
                if action not in ["buy", "sell", "hold"]:
                    validation_errors.add(f"股票的 action 必须是 'buy'、'sell' 或 'hold' 之一")

                # 新增验证：对于在 self.potential_stock_list 的股票，验证 action 是否为 hold 或 buy
                if stock_code in self.potential_stock_list and action not in ["hold", "buy"]:
                    validation_errors.add(f"潜在股票（之间持仓为0） 的 action 必须是 'hold' 或 'buy' 之一")

                # 5. 验证 cur_position
                cur_position = decision_dict.get("cur_position")
                expected_cur_position = cur_positions.get(stock_code, {}).get('ratio', 0)
                if not isinstance(cur_position, (int, float)) or cur_position < 0 or cur_position > 100:
                    validation_errors.add(f"股票的cur_position 必须是 0 到 100 之间的数字")
                elif cur_position != expected_cur_position:
                    validation_errors.add(f"股票的cur_position 必须与提供的持仓信息一致")

                # 6. 验证 target_position
                target_position = decision_dict.get("target_position")
                if not isinstance(target_position, (int, float)) or target_position < 0 or target_position > 100:
                    validation_errors.add(f"股票的target_position 必须是 0 到 100 之间的数字")
                else:
                    total_target_position += target_position

                # 7. 验证 target_price
                target_price = decision_dict.get("target_price")
                if not isinstance(target_price, (int, float)):
                    validation_errors.add(f"股票的 target_price 必须是有效的数字")
                elif action in ["buy", "sell"] and not (price_info[stock_code]['limit_down'] <= target_price <= price_info[stock_code]['limit_up']):
                    validation_errors.add(f"股票的 target_price 必须在跌停价和涨停价之间")

                # 验证 hold 时 target_position 和 target_price 为 0
                if action == "hold":
                    if target_position != cur_position:
                        decision_dict["target_position"] = cur_position

                # 更新决策信息
                stock_decisions[stock_code] = decision_dict

            # 8. 验证 target_positions 之和
            if total_target_position > available_position:
                validation_errors.add("所有股票的 target_position 之和不能超过可用仓位")

        # 如果有验证错误，抛出异常
        if validation_errors:
            formatted_errors = "## 请注意:\n- " + "\n- ".join(validation_errors)
            raise ValueError(formatted_errors)

        return decision_args

    def _read_news(self):
        """
        异步处理导入的新闻列表并获取AI分析结果
        """
        try:
            if not self.import_news or not isinstance(self.import_news, list):
                self.conversation_history.append({
                    "role": "user",
                    "content": "进行信息检索后，我没有找到任何重要到需要群体广播的新闻。"})
                return

            # 确保所有新闻都是字符串格式并去除空值和NaN
            news_list = [str(news) for news in self.import_news if news and not pd.isna(news)]

            # 如果过滤后没有新闻
            if not news_list:
                self.conversation_history.append({
                    "role": "user",
                    "content": "进行信息检索后，我没有找到任何有效的新闻。"})
                return

            # 去重
            news_list = list(dict.fromkeys(news_list))

            # 获取新闻分析提示
            news_prompt = TradingPrompt.get_news_analysis_prompt(news_list)

            # 添加用户提示到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": news_prompt
            })

            # 使用BaseAgent进行异步调用
            news_agent = self.base_agent
            start_time = time.time()

            result = news_agent.get_response(
                messages=self.conversation_history
            )
            print_debug(f"新闻分析耗时: {time.time() - start_time:.2f}秒", self.debug)

            self.news_sumary = result.get("response")
            self.conversation_history.append({
                "role": "assistant",
                "content": self.news_sumary
            })
            # print(result)
        except Exception as e:
            print(f"处理新闻时发生错误: {str(e)}")
            # 可以选择添加错误信息到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": "在处理新闻时遇到了技术问题，暂时无法分析最新新闻。"
            })

    def _polish_decision(self, decision_args: dict, price_info: dict, cur_positions: dict, available_position: float) -> dict:
        '''
        做这样的事情：
（1） 如果action修正之后是hold（要么action是hold，要么trading_position是0），直接把target_price设置成price_info的pre_close，然后新增target_positon和cur_position（从cur_positions获得）并且设置相同，删掉trading_position这个key; 
 (2) 检查trading_position是否有负数，如果有负数，并且action是卖出，直接取绝对值；如果action不是sell，直接变成hold,做前面的相同操作 
（3）如果确定是卖出，看trading_position有没有超过自己的cur_position，如果没有，问题不大，如果超过了，直接设置成trading_position=cur_position,同样新增target_positon（cur-trading)和cur_position，删掉trading_position这个key;,并且更新当前的avail_position;
(4)如果确定是买入，看trading_position有没有超过原始的avail_position，如果小于，先不动，如果大于，变成这个avail_position；
（5）如果股票是self.potential_stock_list，检查是否存在sell，如果有变成hold
(6)如果交易的股票价格不在交易区间内，如果大于就设置成最大值，小于设置成最小值  
（7）全部过滤一遍后，如果存在超过原始avail_position的情况，把所有sell的trading_position累加之后，得到一个sum，用traidng_position/sum*更新的avail_position，向下取证，同样给buy的也新增target_positon和cur_position，删掉trading_position这个key;   你需要根据我的描述，帮我修改
        '''

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
                decision.get("action") if isinstance(decision, dict)
                and decision.get("action") in ["buy", "sell", "hold"]
                else "hold"
            )

            trading_position = (
                float(decision.get("trading_position", 0.0))
                if isinstance(decision, dict)
                and isinstance(decision.get("trading_position"), (int, float))
                else 0.0
            )

            cur_position = cur_positions.get(stock_code, {}).get('ratio', 0)

            # (1) 处理负数 trading_position
            if trading_position < 0:
                if new_decision["action"] == "sell":
                    trading_position = abs(trading_position)
                else:
                    new_decision["action"] = "hold"

            # (2) 处理潜在股票列表
            if stock_code in self.potential_stock_list and new_decision["action"] == "sell":
                new_decision["action"] = "hold"

            # (3) 处理 hold 情况
            if new_decision["action"] == "hold" or trading_position == 0:
                new_decision["action"] = "hold"
                new_decision["target_position"] = cur_position
                new_decision["cur_position"] = cur_position
                new_decision["target_price"] = price_info.get(stock_code, {}).get('pre_close', 0)
                stock_decisions[stock_code] = new_decision
                continue

            # (4) 处理卖出情况
            if new_decision["action"] == "sell":
                if trading_position > cur_position:
                    trading_position = cur_position
                new_decision["target_position"] = cur_position - trading_position
                new_decision["cur_position"] = cur_position
                available_position += trading_position

            # (5) 处理买入情况
            if new_decision["action"] == "buy":
                if trading_position > original_available:
                    trading_position = original_available

            # (6) 处理价格区间
            if stock_code in price_info:
                limit_up = price_info[stock_code].get('limit_up', float('inf'))
                limit_down = price_info[stock_code].get('limit_down', 0)
                pre_close = price_info[stock_code].get('pre_close', 0)

                if new_decision["action"] != "hold":
                    # 生成以昨收为均值,标准差为3%的正态分布随机价格
                    std = pre_close * 0.03
                    random_price = random.normalvariate(pre_close, std)
                    random_price = round(random_price, 2)
                    # 确保价格在涨跌停区间内
                    new_decision["target_price"] = min(max(random_price, limit_down), limit_up)
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
            cur_pos = cur_positions.get(stock_code, {}).get('ratio', 0)

            if decision["action"] == "buy":
                if total_buy_position > available_position:
                    # 如果总买入量超过可用仓位，按比例调整
                    ratio = available_position / total_buy_position
                    adjusted_trading = math.floor(decision["trading_position"] * ratio * 100) / 100
                else:
                    # 如果没超过，使用原始trading_position
                    adjusted_trading = decision["trading_position"]

                decision["target_position"] = cur_pos + adjusted_trading
                decision["cur_position"] = cur_pos
                del decision["trading_position"]

            elif decision["action"] == "sell":
                del decision["trading_position"]

        decision_args["stock_decisions"] = stock_decisions
        return decision_args

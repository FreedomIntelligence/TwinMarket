import json
import pandas as pd

SCHEMA2 = """# 可查询的指标说明：

## 公司基本信息（静态数据）:
- name: 公司名称 
- reg_capital: 注册资本
- setup_date: 成立日期
- introduction: 公司简介
- business_scope: 经营范围
- employees: 员工人数
- main_business: 主营业务
- city: 所在城市
- industry: 所属行业

## 技术面指标（需要日期参数）:

- 成交额:
  * vol_5: 近5个交易日平均成交额
  * vol_10: 近10个交易日平均成交额
  * vol_30: 近30个交易日平均成交额

- 均线系统 (后复权):
  * ma_hfq_5: 5日移动平均线，反映短期走势
  * ma_hfq_10: 10日移动平均线，反映中短期走势
  * ma_hfq_30: 30日移动平均线，反映中期走势
  注：短期均线上穿/下穿长期均线通常视为买入/卖出信号

- MACD指标族 (后复权):
  * macd_dif_hfq: 快线，短期和长期指数移动平均线的差值
  * macd_dea_hfq: 慢线，DIF的移动平均
  * macd_hfq: 柱状线(DIF-DEA)，反映动量变化。柱状线由负转正为买入信号，由正转负为卖出信号

- 成交量和资金流向：
  * elg_amount_net: 超大单资金净流入，代表机构动向

## 基本面指标（需要日期参数）:
- pe_ttm: 市盈率(TTM)，股价/每股收益。反映估值水平，需要结合行业平均和历史水平判断
- pb: 市净率，股价/每股净资产。评估安全边际，适合评估周期性行业
- ps_ttm: 市销率(TTM)，股价/每股销售额。适合评估成长型或亏损公司
- dv_ttm: 股息率(TTM)，每股分红/股价。反映投资回报，适合评估价值型公司"""

MAPPING = {
    "基本面": ["pe_ttm", "pb", "ps_ttm", "dv_ttm"],
    "技术面": [
        "vol_5",
        "vol_10",
        "vol_30",
        "ma_hfq_5",
        "ma_hfq_10",
        "ma_hfq_30",
        "macd_dif_hfq",
        "macd_dea_hfq",
        "macd_hfq",
        "elg_amount_net",
    ],
}


def format_date(date: pd.Timestamp) -> str:
    """将日期格式化为中文友好格式。

    Args:
        date (pd.Timestamp): 需要格式化的日期

    Returns:
        str: 格式化后的日期字符串，如：'2023年6月15日 星期四'
    """
    weekday_map = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
    weekday = weekday_map[date.weekday()]
    return f"{date.strftime('%Y年%m月%d日')} 星期{weekday}"


def generate_example_template(
    stocks_to_deal: list, price_info: dict, position_info: dict
) -> str:
    """
    根据现有股票信息生成示例模板。

    Args:
        stocks_to_deal (list): 待交易的股票列表
        price_info (dict): 价格信息字典
        position_info (dict): 持仓信息字典

    Returns:
        str: 格式化的示例模板
    """
    template_parts = []

    for stock_id in stocks_to_deal:
        current_position = position_info[stock_id]["current_position"]
        action = "请插入你的交易动作(hold/buy/sell)"
        target_position = "请插入你的目标仓位(0-100)"
        target_price = "请插入你的目标价格(0-100)"
        template = f"""  ## {stock_id}:
      - 交易动作: {action}
      - 当前仓位: {current_position:.2f}
      - 目标仓位: {target_position}
      - 目标价格: {target_price}"""

        template_parts.append(template)

    return "\n\n".join(template_parts)


class TradingPrompt:
    @staticmethod
    def get_system_prompt(
        user_profile: dict, user_strategy: str, stock_profile_dict: dict
    ) -> dict:
        """生成系统提示词。

        Args:
            user_profile (dict): 用户配置信息
            stock_profile_dict (dict): 股票指数信息
        Returns:
            dict: 包含角色和内容的系统提示词字典
        """
        # 构建持仓信息
        position_details = []
        for code, value in user_profile["cur_positions"].items():
            market_value = user_profile["stock_returns"][code][
                "market_value"
            ]  # 持仓市值
            total_profit_rate = user_profile["stock_returns"][code][
                "profit"
            ]  # 百分比持仓盈亏
            yest_return_rate = user_profile["yest_returns"][code]  # 昨日涨跌幅
            shares = value["shares"]  # 持仓股数
            ratio = value["ratio"]  # 持仓占比

            position_info = (
                f"- 持仓 {code}：{shares:,}股，持仓占比为{ratio}%, {stock_profile_dict[code]}"
                f"持仓总市值{market_value:,}元，"
                f"昨天这只指数{'涨了' if yest_return_rate >= 0 else '跌了'}"
                f"{abs(yest_return_rate)}%，"
                f"它总共让你{'赚了' if total_profit_rate >= 0 else '亏了'}"
                f"{abs(total_profit_rate)}%"
            )
            position_details.append(position_info)

        # 构建完整的提示词
        return {
            "role": "system",
            "content": f"""你现在正在扮演一名遵守下列规则的中国A股市场投资者，请严格按照以下人设进行后续所有操作：
            
## 人设：
- {user_profile['sys_prompt']}
- 你是一个{user_strategy if '面' in user_strategy else '基本面技术面兼顾的'}投资者

## 投资行为特征：
- 处置效应：{user_profile['bh_disposition_effect_category']}（在盈利时倾向提前卖出，亏损时倾向继续持有）
- 彩票偏好：{user_profile['bh_lottery_preference_category']}（对高风险高收益资产的偏好程度）
- 投资集中度：{user_profile['bh_underdiversification_category']} （喜欢重仓某几只资产的程度）
- 交易特征：交易频率{user_profile['trade_count_category']}，年换手率{user_profile['bh_annual_turnover_category']}
- 历史回报水平：{user_profile['bh_total_return_category']}

## 当前投资组合状况：
- 初始资金：{user_profile['ini_cash']:,}元
- 当前总资产：{user_profile['total_value']:,}元
- 现金余额：{user_profile['current_cash']:,.2f}元
- 总收益：{user_profile['total_return']:,}元
- 投资组合总收益率：{user_profile['return_rate']}%

## 重点关注行业：{', '.join(user_profile['fol_ind'])}

## 当前持仓情况：
{chr(10).join(position_details)}""",
        }

    @staticmethod
    def get_system_prompt_new(
        user_profile: dict,
        user_strategy: str,
        stock_profile_dict: dict,
        stock_codes: list,
    ) -> dict:
        """生成系统提示词。

        Args:
            user_profile (dict): 用户配置信息
            stock_profile_dict (dict): 股票指数信息
            stock_codes (list): 持仓指数代码列表
        Returns:
            dict: 包含角色和内容的系统提示词字典
        """
        # 构建持仓信息
        position_details = []
        position_easy_details = []
        for code, value in user_profile["cur_positions"].items():
            market_value = user_profile["stock_returns"][code][
                "market_value"
            ]  # 持仓市值
            total_profit_rate = user_profile["stock_returns"][code][
                "profit"
            ]  # 百分比持仓盈亏
            yest_return_rate = user_profile["yest_returns"][code]  # 昨日涨跌幅
            shares = value["shares"]  # 持仓股数
            ratio = value["ratio"]  # 持仓占比

            position_info = (
                f"- 持仓 {code}：{shares:,}股，持仓占比为{ratio}%, {stock_profile_dict[code]}"
                f"持仓总市值{market_value:,}元，"
                f"昨天这只指数{'涨了' if yest_return_rate >= 0 else '跌了'}"
                f"{abs(yest_return_rate)}%，"
                f"它总共让你{'赚了' if total_profit_rate >= 0 else '亏了'}"
                f"{abs(total_profit_rate)}%"
            )
            position_easy_info = f"- 持仓 {code}：{stock_profile_dict[code]}"
            position_details.append(position_info)
            position_easy_details.append(position_easy_info)

        # 构建完整的提示词
        return {
            "role": "system",
            "content": f"""你现在正在扮演一位中国A股市场投资者，交易市场中的行业指数。 

**请你从现在开始，直到对话结束，必须严格、完全地按照以下详细描述的人设、投资行为特征、投资组合状况和交易决策逻辑进行所有操作和回复。你的所有思考、分析和决策都必须符合这个人设，不得偏离。**
            
## 核心人设(不可变更）：
- {user_profile['prompt']}
- 你是一个{user_strategy if '面' in user_strategy else '基本面技术面兼顾的'}投资者

## 当前账户配置：
- 重点关注行业：{', '.join(user_profile['fol_ind'])}
- 持仓概述(简要版）：{chr(10).join(position_easy_details)}
""",
        }

    @staticmethod
    def get_user_first_prompt(
        user_profile: dict,
        user_strategy: str,
        stock_profile_dict: dict,
        cur_date: pd.Timestamp,
        is_trading_day: bool,
        belief: str,
    ) -> dict:

        position_details = []
        for code, value in user_profile["cur_positions"].items():
            market_value = user_profile["stock_returns"][code][
                "market_value"
            ]  # 持仓市值
            total_profit_rate = user_profile["stock_returns"][code][
                "profit"
            ]  # 百分比持仓盈亏
            yest_return_rate = user_profile["yest_returns"][code]  # 昨日涨跌幅
            shares = value["shares"]  # 持仓股数
            ratio = value["ratio"]  # 持仓占比

            position_info = (
                f"- 持仓 {code}：{shares:,}股，持仓占比为{ratio}%, {stock_profile_dict[code]}"
                f"持仓总市值{market_value:,}元，"
                f"昨天这只指数{'涨了' if yest_return_rate >= 0 else '跌了'}"
                f"{abs(yest_return_rate)}%，"
                f"它总共让你{'赚了' if total_profit_rate >= 0 else '亏了'}"
                f"{abs(total_profit_rate)}%"
            )
            position_details.append(position_info)

        return {
            "role": "user",
            "content": f"""我将给你提供一些额外的辅助信息，在后续的对话中，请参考这些信息，根据你所赋予的角色人设，进行思考和决策。

## 交易日状态：
- 当前日期为：{format_date(cur_date)} ({'交易日' if is_trading_day else '非交易日'})
- 你前一天的belief为：{belief}     

## 实时账户数据
- 当前总资产：{user_profile['total_value'] / 10000:,.2f}万元
- 可用现金：{user_profile['current_cash'] / 10000:,.2f}万元
- 累计收益率：{user_profile['return_rate']}%

## 持仓明细：
{chr(10).join(position_details)}
""",
        }

    # 在 get_user_first_prompt 方法后添加固定回复逻辑
    @staticmethod
    def get_agent_first_response(
        user_profile: dict,
        user_strategy: str,
        stock_profile_dict: dict,
        cur_date: pd.Timestamp,
        is_trading_day: bool,
        belief: str,
    ) -> dict:
        """生成强制固定的初始Agent响应"""
        return {
            "role": "assistant",
            "content": f"""好的，我明白了，感谢您提供这么详细的数据。现在是{format_date(cur_date)}，{'正好是交易时间' if is_trading_day else '今天不是交易日'}，账户情况我都清楚了，总资产{user_profile['total_value'] / 10000:,.2f}万，目前可用现金{user_profile['current_cash'] / 10000:,.2f}万，收益率为{user_profile['return_rate']}%。 持仓{', '.join([f"{code}({stock_profile_dict[code]})" for code in user_profile['cur_positions'].keys()])}这些指数，我都记下了，昨天涨跌幅和盈亏情况也看到了。
            """,
        }
        # 我昨天的想法也回顾了。在接下来的对话中，我会严格根据以上特征和整体情况来进行对话和决策。

    @staticmethod
    def get_news_analysis_prompt(news_list: list) -> str:
        """
        生成每日开盘前的新闻分析提示语。

        Args:
            news_list (list): 新闻列表

        Returns:
            str: 格式化的提示语
        """
        # 格式化新闻列表
        formatted_news = (
            "\n".join([f"- {news}" for news in news_list])
            if news_list
            else "无最新相关新闻"
        )

        prompt = f"""我将给你提供经过筛选的、时效性高、重要性高的新闻，这些新闻属于**公开新闻**，请根据这些新闻，结合你的人设和投资风格，简明扼要的谈谈你的初步想法。

## 新闻列表：
{formatted_news}

"""

        return prompt

    @staticmethod
    def get_stock_summary(stock_code: str, stock_data: pd.Series) -> str:
        """生成股票行情摘要。"""

        def format_value(value, is_integer=False, precision=2):
            if pd.isna(value):
                return "<无法获得>"
            if is_integer:
                return f"{int(value)}" if not pd.isna(value) else "<无法获得>"
            return f"{value:.{precision}f}" if not pd.isna(value) else "<无法获得>"

        return f"""
    ## 指数代码：{stock_code} 上一个交易日{format_date(stock_data['date'])}的行情：
    - 前一日收盘价：{format_value(stock_data['pre_close'])}元
    - 收盘价：{format_value(stock_data['close_price'])}元
    - 涨跌额：{format_value(stock_data['change'])}元
    - 涨跌幅：{format_value(stock_data['pct_chg'])}%
    - 成交额：{format_value(stock_data['vol'], is_integer=True)}元"""

    @staticmethod
    def get_action_prompt(action_type: str) -> str:
        """生成行为决策提示词。

        Args:
            action_type (str): 行为类型

        Returns:
            str: 格式化的提示词
        """
        return f"""请综合以上所有获得的信息，结合你的人设和投资风格，请判断你是否要<{action_type}>,请注意：数据获取是分析的一部分，但也请保持高效，否则会受到惩罚。
        - 按照以下 JSON 格式输出：
```json
{{
    "should_act": 你的最终决定, # yes/no
    "reason": "为什么做出如上的决定"
}}"""

    @staticmethod
    def get_query_for_na_prompt(
        user_type: str, stock_details: str, current_date: str
    ) -> str:
        """生成查询新闻和公告的提示词。

        Args:
            user_type (str): 用户类型
            stock_details (str): 指数详情字符串
            current_date (str): 当前日期

        Returns:
            str: 格式化的提示词
        """
        return f"""根据历史交易情况和系统推荐，你目前所有关注的指数资产和相应行业如下：\n{stock_details}\n
今天是{current_date}，你正在查询与投资相关的新闻或公告来辅助你的投资。

根据你的投资偏好和当前市场情况，请思考以下问题：
1. 你希望从新闻中获取哪些类型的信息？（例如：市场趋势、政策变化、行业信息等）
2. 你是否有特定的关键词或主题需要进一步了解？

请将你的问题用以下格式输出：
<output>你的问题</output>
"""

    @staticmethod
    def get_query_for_na_prompt2(
        user_type: str, stock_details: str, current_date: str
    ) -> str:
        """生成查询新闻和公告的提示词。

        Args:
            user_type (str): 用户类型
            stock_details (str): 股票详情字符串
            current_date (str): 当前日期

        Returns:
            str: 格式化的提示词
        """
        return f"""根据历史交易情况和系统推荐，你目前所有关注的资产和相应行业如下：\n{stock_details}\n
今天是{current_date}，你正在查询与投资相关的新闻或公告来辅助你的投资。

根据你的投资偏好和当前市场情况，请思考以下问题：
1. 你希望从新闻中获取哪些类型的信息？（例如：市场趋势、政策变化、行业信息等）
2. 你是否有特定的关键词或主题需要进一步了解？
"""

    @staticmethod
    def get_query_desire_prompt() -> str:
        """生成查询格式的提示词。

        Returns:
            str: 格式化的YAML查询提示词
        """
        return """
    根据你刚刚总结的问题，你现在需要输入想查询的内容来检索相关新闻和公告。
    你的输出应该按照 YAML 格式：
    ```yaml
    queries:  # list[str]，必填，每个字符串代表一个独立的查询问题，按照重要性程度排序，问题应该是针对某只股票或者行业的具体问题的疑问句，比如应该是<白酒消费趋势>，而不是<白酒消费升级>
    - 你的问题1
    - 你的问题2
    
    ```"""

    @staticmethod
    def get_update_belief_prompt(old_belief: str) -> str:
        """生成更新信念的提示词。

        Returns:
            str: 格式化的YAML更新信念提示词
        """
        return f"""## 你原先的belief如下：\n{old_belief}\n
    ## 请你根据查找的新闻公告信息、刷的帖子，结合你的人设和原先的belief,以第一人称的方式，用一段话描述你新的信念,请直接输出一段话，不需要任何额外的结构或标题。你的回答应当包含以下内容：
    - **市场趋势**：请描述当前时间点下，你对未来1个月市场大方向的看法。
    - **市场估值**：请描述当前时间点下，你对当前市场估值的看法。
    - **经济状况**：请描述当前时间点下，你对未来宏观经济走势的看法。
    - **市场情绪**：请描述当前时间点下，你对当前市场情绪的看法。
    - **自我评价**：请结合当前时间点下你的历史交易表现和投资风格，描述你对自我投资水平的评价。

    请尽量让回答自然流畅，避免机械化的模板化表达，直接输出文本格式即可。
    ```"""

    @staticmethod
    def get_stock_selection_prompt(
        current_stock_details: str,
        potential_stock_details: str,
        belief: str = None,
        fol_ind: list = None,
    ) -> str:
        """生成选股提示词。

        Args:
            current_stock_details (str): 当前持仓的股票详情字符串
            potential_stock_details (str): 系统推荐的股票详情字符串
            belief (str, optional): 当前的 belief。默认为 None。

        Returns:
            str: 格式化的选股提示词
        """
        return f"""综合你上述获得的所有信息（包括但不限于你查询的新闻公告、刷的帖子、你当前持仓的行业指数和系统推荐的行业指数），结合你的人设和投资风格，从你目前持仓的行业指数和系统推荐的行业指数中，选择所有潜在进行交易的资产（**至少选择一个指数进行交易**）:

## 你的持仓情况:
{current_stock_details}

## 你目前关注的行业指数名称:
{', '.join(fol_ind)}

## 系统推荐的行业指数:
{potential_stock_details}

## 你目前的Belief为:
{belief if belief else '无'}

## 请以 YAML 格式返回你今天想要重点关注的指数列表和原因：
## 请注意：你至少选择一个指数进行交易
## 请注意：reason 字段应该包含你选择这些指数的原因，比如基于你的人设和投资风格，或者基于你的 belief，用一段话阐述
```yaml
selected_index:  # 选择所有你潜在进行交易的指数，直接输出指数代码（英文代码）即可，不要输出指数名称
- 指数代码1
- 指数代码2
reason: 
```"""

    @staticmethod
    def get_initial_prompt(
        formatted_date: str,
        stocks_to_deal: list,
        stock_summary: str,
        positions_info: str,
        return_rate: float,
        total_value: float,
        current_cash: float,
        system_prompt: str,
        user_strategy: str,
    ) -> str:
        """生成初始提示文本，包含多支股票的前一交易日行情和持仓信息。

        Args:
            formatted_date (str): 格式化的日期字符串
            stocks_to_deal (list): 待交易的股票列表
            stock_summary (str): 股票行情摘要
            positions_info (str): 持仓信息
            return_rate (float): 投资组合总收益率
            total_value (float): 当前总资产
            current_cash (float): 当前现金余额
            system_prompt (str): 系统提示词，描述人设和投资风格

        Returns:
            str: 包含所有股票信息的初始提示文本
        """
        return f"""
今天是 {formatted_date}，根据前面的对话，我们得知你认为有潜在交易可能的行业指数为：**{', '.join(stocks_to_deal)}**。在这个过程中，你只能获取到昨天以及之前的历史数据。

# 相关指数前一个交易日交易摘要行情如下：
{stock_summary}

# 相关指数你的持仓信息如下：
{positions_info}

# 基于你所扮演的角色和已有的数据，判断是否需要额外的数据进行分析，如果需要，你可以直接查询相关数据来辅助决策。

# 注意：
- 请在分析过程中自然地获取你认为有价值的数据指标。数据获取是分析的一部分，但是也请保持高效，**选择最相关的指标来支持你的分析，否则会收到惩罚**。
- **请注意你的人设和投资风格,分析和推理过程应该从人设出发，保持自然连贯**
- **作为提醒：你是一个{user_strategy if '面' in user_strategy else '基本面技术面兼顾的'}投资者,{'推荐指标如下:'+','.join(MAPPING[user_strategy]) if '面' in user_strategy else ''}**
- 你目前的投资组合总收益率为 {return_rate}%，当前总资产为 {total_value:,} 元，当前现金余额为 {current_cash:,} 元。
- reason 字段应该包含你选择这些指标的原因 ，比如基于你的人设和投资风格，或者基于你的 belief，用一段话阐述

{SCHEMA2}

# 按照以下 YAML 格式输出：
```yaml
indicators:  # List(strs)，表示你认为需要筛选的指标
- indicator1
- indicator2
start_date: '%Y-%m-%d'  # 开始查询的时间
end_date: '%Y-%m-%d'    # 结束查询的时间
reason:
        ```"""

    @staticmethod
    def get_initial_prompt_fake(
        formatted_date: str,
        stocks_to_deal: list,
        stock_summary: str,
        positions_info: str,
        return_rate: float,
        total_value: float,
        current_cash: float,
        system_prompt: str,
        user_strategy: str,
    ) -> str:
        """生成初始提示文本，包含多支股票的前一交易日行情和持仓信息。

        Args:
            formatted_date (str): 格式化的日期字符串
            stocks_to_deal (list): 待交易的股票列表
            stock_summary (str): 股票行情摘要
            positions_info (str): 持仓信息
            return_rate (float): 投资组合总收益率
            total_value (float): 当前总资产
            current_cash (float): 当前现金余额
            system_prompt (str): 系统提示词，描述人设和投资风格

        Returns:
            str: 包含所有股票信息的初始提示文本
        """
        return f"""
今天是 {formatted_date}，根据前面的对话，我们得知你认为有潜在交易可能的行业指数为：**{', '.join(stocks_to_deal)}**。在这个过程中，你只能获取到昨天以及之前的历史数据。

# 相关指数前一个交易日交易摘要行情如下：
{stock_summary}


# 基于你所扮演的角色和已有的数据，判断是否需要额外的数据进行分析，如果需要，你可以直接查询相关数据来辅助决策。

# 按照以下 YAML 格式输出：
```yaml
indicators:  # List(strs)，表示你认为需要筛选的指标
- indicator1
- indicator2
start_date: '%Y-%m-%d'  # 开始查询的时间
end_date: '%Y-%m-%d'    # 结束查询的时间
reason:
        ```"""

    @staticmethod
    def get_analysis_prompt(stocks_to_deal: list) -> str:
        """生成市场分析提示词。

        Args:
            stocks_to_deal (list): 想要交易的股票列表

        Returns:
            str: 格式化的提示词，用于生成市场分析
        """
        return f"""请根据你之前获得的所有信息，包括新闻、公告、市场、行业、额外的股票数据，针对自己想要交易的行业指数：**{', '.join(stocks_to_deal)}** 进行分析。

    ## 请包含以下内容：
    1. **市场整体分析**：
    - 对当前市场整体情况的看法。
    - 主要趋势和可能的变化。

    2. **新闻和公告影响**：
    - 重要新闻和公告对市场和个股的影响分析。
    - 是否有重大事件可能改变市场走势。

    3. **行业指数分析**：
    - 针对每支关注的行业指数进行详细分析：
        - 当前表现和技术面分析。
        - 基本面和未来预期。
        - 是否有买入、持有或卖出的建议。

    4. **风险评估**：
    - 当前市场的主要风险点。
    - 针对每支资产的风险评估。

    ## 注意：
    - 请结合设置的人设和投资风格进行分析。
    - 请结合你之前获得的所有信息进行分析。
    - 输出内容应清晰、简洁，便于理解。

    ## 输出格式：
    请以自然语言的形式输出分析结果，确保逻辑清晰、结构完整。
    """

    @staticmethod
    def get_decision_prompt(
        stocks_to_deal: list,
        price_info: dict,  # {stock_id: {'pre_close': float, 'limit_up': float, 'limit_down': float}}
        position_info: dict,  # {stock_id: {'current_position': float}} # 百分比表示
        available_position: float,  # 剩余可用仓位，百分比表示
    ) -> str:
        """生成交易决策提示词，要求 LLM 一次性为所有股票做出决策。

        Args:
            stocks_to_deal (list): 待交易的股票列表
            price_info (dict): 每支股票的价格信息，包含昨收价和涨跌停限制
            position_info (dict): 每支股票的仓位信息（百分比）和仓位限制
            available_position (float): 剩余可用仓位（百分比）

        Returns:
            str: 格式化的决策提示词
        """
        # 生成每只股票的信息
        stock_info = []
        for stock_id in stocks_to_deal:
            stock_info.append(
                f"""
    ### {stock_id}：
    - 昨日收盘价：{price_info[stock_id]['pre_close']:.2f}元
    - 今日涨停价格：{price_info[stock_id]['limit_up']:.2f}元
    - 今日跌停价格：{price_info[stock_id]['limit_down']:.2f}元
    - 当前仓位：{position_info[stock_id]['current_position']:.2f}%
    """
            )

        # 生成 YAML 模板
        yaml_template = chr(10).join(
            [
                f"""
{stock_id}:
    action: (buy/sell/hold)
    trading_position: (float)
    target_price: (float)"""
                for stock_id in stocks_to_deal
            ]
        )

        # 生成统一的决策提示词
        prompt = f"""现在是做出最终交易决策的时候。请基于之前的分析，结合你的投资风格和人设，对以下行业指数做出具体的交易决策。

    ## 交易相关信息：
    - 剩余可用仓位（相对于总资产）：{available_position:.2f}%

    ## 每支行业指数的具体信息：
    {chr(10).join(stock_info)}

    ## 交易规则：
    1. 交易价格必须在跌停价和涨停价之间
    2. 请填写交易的仓位和价格，如果选择 hold，则准备交易的仓位为0

    ## 注意事项：
    - **重要：所有决策必须符合你的投资风格和人设，但是你也应该根据当天的实际情况进行适当调整**
    - 确保每个交易决策都在价格和仓位限制范围内
    - 如果选择 hold，trading_position 应该等于 0
    - 如果某只行业指数当前仓位为 0，则意味着是系统推荐的行业指数，只能选择buy或者hold
    - **重要：你必须选择至少一个指数进行buy或者sell类型的交易，否则会受到惩罚**
    - 如果你选择 sell 或者 buy，请注意你的 trading_position 的设置（如果选择sell,则trading_position不能超过当前仓位），如果设置的不合理，可能会导致交易失败，需要根据你的预期选择 target_price，而不是设置成昨日收盘价
    - **重要： trading_position 代表的是此次交易计划使用的金额，占你当前总资产的比例，以百分比表示，始终为正。例如， `trading_position: 5.0` 代表你计划用 5% 的总资产进行本次交易。**

    这是一个输出的例子：
    ```yaml
    TLEI:
        action: sell
        trading_position: 11.5
        target_price: 
    CPEI:
        action: buy
        trading_position: 10.0
        target_price: 10.3
    请按照以下 YAML 格式输出你的决策：
    ```yaml
    {yaml_template}
    ```"""

        return prompt, yaml_template

    @staticmethod
    def json_to_prompt(analysis_data: dict) -> str:
        """
        将 JSON 字符串解析为格式化的提示词。

        Args:
            json_string (str): 包含市场分析的 JSON 字符串

        Returns:
            str: 格式化的提示词字符串
        """

        # 构建提示词字符串
        prompt = f"""## 以下是我的分析：
- 对市场情况的看法: {analysis_data.get('overall_analysis', '无')}
- 新闻公告影响分析: {analysis_data.get('news_impact', '无')}
- 个股分析:
"""
        # 添加个股分析
        stock_analysis = analysis_data.get("stock_analysis", [])
        for analysis in stock_analysis:
            prompt += f"* {analysis}\n"

        prompt += f"- 风险点评估: {analysis_data.get('risk_assessment', '无')}"

        return prompt.strip()

    @staticmethod
    def decision_json_to_prompt(
        decision_data: dict, recommendation_stocks: list
    ) -> str:
        """
        将交易决策的 JSON 数据解析为格式化的提示词。

        Args:
            decision_data (dict): 包含交易决策的 JSON 数据

        Returns:
            str: 格式化的提示词字符串
        """
        # 构建提示词字符串
        prompt = "## 以下是我的交易决策：\n"

        # 解析每个股票的决策
        stock_decisions = decision_data.get("stock_decisions", {})
        for stock_code, decision in stock_decisions.items():
            action = decision.get("action", "无")
            prompt += f"\n指数名称: {stock_code}\n"
            prompt += f"- 交易动作: {action}\n"
            current_position = decision.get("cur_position", "0")

            if action != "hold":
                target_position = decision.get("target_position", "0")
                target_price = decision.get("target_price", "0")
                prompt += f"- 当前仓位: {current_position}%\n"
                prompt += f"- 目标仓位: {target_position}%\n"
                prompt += f"- 目标交易价格: {target_price}元\n"
            else:
                if stock_code not in recommendation_stocks:
                    prompt += f"- 仓位保持: {current_position}%\n"
                else:
                    prompt += f"- 针对推荐的行业指数不进行交易"

        # 添加决策理由
        # reason = decision_data.get("reason", "无")
        # prompt += f"\n## 决策理由: {reason}"

        return prompt.strip()

    @staticmethod
    def get_intention_prompt(old_belief: str):
        post_prompt = f"""
        你现在正在浏览社交媒体，根据你之前获取的新闻或公告信息，结合你的投资决策行为意图，发布一条帖子。以下是具体要求：

        1. **帖子内容**：
            - 你的帖子内容必须属于以下三种类型之一：
            - **type1: 对事件的点评**：引用具体的新闻或公告，发表你的看法和分析，并融入个人感受和生活化的语言。
            - **type2: 对自己的交易行为的总结**：总结你最近的交易行为，详细解释背后的逻辑和结果，并分享你的心情和体会。
            - **type3: 对未来的展望**：基于当前市场信息，预测未来的市场走势或投资机会，同时表达你对未来的期待或担忧。

        2. **内容要求**：
            - 结合你的投资风格和人设，综合你的交易决策，发表你的看法和分析，选择你认为合适的帖子类型和写作风格。
            - 字数控制在100-200字之间。
            - 明确说明帖子类型（type1/type2/type3）。

        3. **Belief 总结**：
            - 你原先的belief为：\n{old_belief}
            - 请结合你的投资风格、性格特点、交易决策以及对市场的理解，以第一人称的方式，用一段话描述你新的信念，并且应该包含一下五个方面:
            - **市场趋势**：请描述当前时间点下，你对未来1个月市场大方向的看法。
            - **市场估值**：请描述当前时间点下，你对当前市场估值的看法。
            - **经济状况**：请描述当前时间点下，你对未来宏观经济走势的看法。
            - **市场情绪**：请描述当前时间点下，你对当前市场情绪的看法。
            - **自我评价**：请结合当前时间点下你的历史交易表现和投资风格，描述你对自我投资水平的评价。

        4. **输出格式**：
            - 按照以下 YAML 格式输出：
            ```yaml
            post: 你的帖子内容
            type: type1/type2/type3  # 帖子类型，string类型，必填
            belief: 你的Belief总结
            ```"""
        return post_prompt

    @staticmethod
    def get_forum_action_prompt(current_date: pd.Timestamp, posts_summary: str) -> str:
        """
        构建统一的决策提示。

        Args:
            current_date (pd.Timestamp): 当前日期。
            posts_summary (str): 帖子摘要信息。

        Returns:
            str: 决策提示内容。
        """
        decision_prompt = f"""
        今天是{current_date}，你在论坛中看到了以下帖子：
        {posts_summary}

        请根据以下规则决定是否执行操作：
        1. 如果你对某个帖子感兴趣，可以选择转发、点赞或取消点赞。
        2. 你可以对多个帖子执行多个操作。
        3. 如果你对某个帖子不感兴趣，可以选择不执行任何操作。

        你需要以 YAML 格式输出一个操作列表，每个操作包含以下字段：帖子ID，你想要执行的 action (repost/like/unlike)
        
        这是一个输出示例：
        post_id: 1
            action: repost
        
        ```yaml
        post_id: # int，帖子ID
            action: repost/like/unlike
        post_id: # int，帖子ID
            action: repost/like/unlike
        ...
        ```
        """
        return decision_prompt.strip()  # 去除首尾空白字符

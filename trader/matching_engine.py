"""
多股票交易撮合引擎系统

该模块实现了一个完整的股票交易撮合引擎，模拟真实证券交易所的订单撮合机制。
系统能够处理大规模用户的交易决策，执行智能订单匹配，并维护完整的交易记录。

核心功能：
1. 智能订单撮合：接收用户交易决策，转换为标准订单格式
2. 多股票并行处理：按股票分组进行高效的订单撮合
3. 完整交易记录：生成详细的交易记录和统计信息
4. 实时数据更新：更新数据库中的股票数据、技术指标和估值指标
5. 大单资金流分析：追踪和分析大额资金流向
6. 可视化支持：生成订单簿可视化图表

主要步骤：

1. 交易决策转换为订单：
   - 输入：用户交易决策列表，每个决策包含：
     * user_id: str (用户ID)
     * stock_code: str (股票代码)
     * direction: str ('buy'/'sell')
     * amount: int (交易数量)
     * target_price: float (目标价格)
   - 处理：
     * 为每个订单随机分配交易时间（9:30-11:30, 13:00-15:00）
     * 确保时间戳不重复
     * 转换为标准Order对象

2. 订单撮合规则：
   - 按股票代码分组处理订单
   - 价格优先：买方出价高者优先，卖方要价低者优先
   - 时间优先：同价格下，先下单者优先
   - 最大成交量原则：在符合价格的订单中寻找最大成交量
   - 涨跌停限制：成交价必须在涨跌停范围内（上一日收盘价±10%）

3. 输出结果：
   a) daily_summary_{date}.csv：每日交易汇总
      - date: 交易日期
      - stock_code: 股票代码
      - closing_price: 收盘价
      - volume: 成交量
      - transaction_count: 成交笔数
      - large_order_net_inflow: 大单(≥100万)资金净流入

   b) transactions_{date}.csv：详细成交记录
      - stock_code: 股票代码
      - user_id: 交易用户ID
      - direction: 交易方向(buy/sell)
      - executed_price: 成交价格
      - executed_quantity: 成交数量
      - original_quantity: 原始委托数量
      - unfilled_quantity: 未成交数量
      - timestamp: 成交时间

   c) large_order_flow_{date}.csv：大单资金流向
      - date: 交易日期
      - stock_code: 股票代码
      - large_order_net_inflow: 大单资金净流入

   d) order_book_{date}.png：订单簿可视化图表

4. 更新数据库：
   更新StockData表中的技术指标、交易数据和估值指标：
   - close_price: 当日收盘价
   - pre_close: 昨日收盘价
   - change: 股价变动
   - pct_chg: 股价涨跌幅(%)
   - pe_ttm: 市盈率(TTM)，根据当日真实数据等比例调整
   - pb: 市净率，根据当日真实数据等比例调整
   - ps_ttm: 市销率(TTM)，根据当日真实数据等比例调整
   - dv_ttm: 股息率(TTM)，根据当日真实数据反比例调整
   - vol: 当日成交量
   - vol_5/10/30: 成交量的5/10/30日均线
   - ma_hfq_5/10/30: 收盘价的5/10/30日均线
   - macd_hfq: MACD柱状线
   - macd_dea_hfq: MACD的DEA线
   - macd_dif_hfq: MACD的DIF线
   - elg_amount_net: 大单资金净流入

数据源说明：
- 历史数据来源：stock_data.csv
- 用途：
  1. 计算技术指标（MA、MACD等）
  2. 获取当日真实交易数据作为基准计算估值指标
  3. 计算移动平均线和成交量均线

使用示例:
准备输入数据
decisions = [
{
'user_id': 'user_001',
'stock_code': 'SH600000',
'direction': 'buy',
'amount': 1000,
'target_price': 10.5
},
# ... 更多交易决策
]
上一日收盘价数据
last_prices = {
'SH600000': 10.0,
# ... 更多股票的收盘价
}
当前日期
current_date = '2023-06-15'
处理交易日数据
results = process_trading_day(decisions, last_prices, current_date)

注意事项：
1. 所有交易数量必须为正整数
2. 交易价格必须在涨跌停范围内（上一日收盘价±10%）
3. 时间戳不允许重复
4. 输出目录默认为 'simulation_results'
5. 大单判定标准为单笔成交金额≥100万
6. 股票代码必须包含市场前缀（如'SH'）
7. 日期格式必须为'YYYY-MM-DD'
8. 技术指标计算需要历史数据支持
9. 成交量均线在数据不足时使用已有数据计算
10. 估值指标根据当日真实数据和模拟价格的比例进行调整
"""

# ============================ 开发状态和待办事项 ============================
# TODO: 修改transactions表格 --user_id  ✅ 已完成
# TODO: 更新Tradingdetails表  ✅ 已完成
# TODO: 验证StockData表更新是否正确 -- 目前写入正确，需要有较多订单验证撮合逻辑
# TODO: 更新user_profile ⏳ 进行中
#
# 调试说明：直接运行debug模式即可测试

# ============================ 导入依赖库 ============================

# 标准库导入
import json
import os
import random
import sqlite3
import uuid
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

# 第三方库导入
import aiofiles  # 异步文件操作
import aiosqlite  # 异步SQLite操作
import matplotlib.pyplot as plt  # 图表绘制
import numpy as np  # 数值计算
import pandas as pd  # 数据处理

# 忽略pandas的FutureWarning警告
warnings.filterwarnings("ignore", category=FutureWarning)


class Order:
    """
    交易订单类

    该类表示一个标准的股票交易订单，包含了订单撮合所需的所有基本信息。
    支持买入和卖出两种方向，并提供时间戳调整功能以避免订单冲突。

    Attributes:
        stock_code (str): 股票代码，如'SH600000'
        price (float): 订单价格
        quantity (int): 订单数量（统一使用正数）
        timestamp (datetime): 订单时间戳
        original_timestamp (datetime): 原始时间戳（备份）
        user_id (str): 下单用户ID
        direction (str): 交易方向，'buy'表示买入，'sell'表示卖出

    Methods:
        adjust_timestamp: 调整订单时间戳以避免重复
        __str__: 订单的字符串表示
        __repr__: 订单的详细表示

    Note:
        - 所有数量都使用正数表示，方向通过direction字段区分
        - 时间戳支持微秒级精度以确保订单优先级
        - 支持时间戳调整以处理并发订单的时间冲突
    """

    def __init__(
        self,
        stock_code: str,
        price: float,
        quantity: int,
        timestamp: datetime,
        user_id: str,
        direction: str,
    ):
        self.stock_code = stock_code
        self.price = price
        self.quantity = quantity  # 统一使用正数
        self.timestamp = timestamp
        self.original_timestamp = timestamp
        self.user_id = user_id
        self.direction = direction  # 'buy' 或 'sell'

    def adjust_timestamp(self, delta_microseconds: int = 1000):
        """
        调整订单时间戳以避免重复

        当多个订单具有相同时间戳时，通过微调时间戳确保订单的唯一性和正确排序。

        Args:
            delta_microseconds (int): 时间戳调整的微秒数，默认1000微秒

        Returns:
            datetime: 调整后的新时间戳
        """
        self.timestamp += timedelta(microseconds=delta_microseconds)
        return self.timestamp

    def __str__(self):
        """订单的简洁字符串表示"""
        return (
            f"Order(price={self.price}, quantity={self.quantity}, "
            f"time={self.timestamp.strftime('%H:%M:%S.%f')})"
        )

    def __repr__(self):
        """订单的详细字符串表示"""
        return self.__str__()


def validate_order_timestamps(orders: list[Order]) -> bool:
    """
    验证订单时间戳的唯一性和合理性

    检查订单列表中是否存在重复的时间戳，确保订单排序的准确性。

    Args:
        orders (list[Order]): 要验证的订单列表

    Returns:
        bool: 如果所有时间戳都唯一则返回True，否则返回False

    Note:
        - 重复时间戳会影响订单的优先级排序
        - 建议在订单处理前进行验证
    """
    timestamps = [order.timestamp for order in orders]

    # 检查是否有重复时间戳
    if len(set(timestamps)) != len(timestamps):
        print("警告: 存在重复的时间戳")
        return False

    return True


def calculate_closing_price(
    buy_orders: list[Order],
    sell_orders: list[Order],
    last_price: float,
    current_date: str = None,
    output_dir: str = None,
) -> tuple[float, int, list[dict]]:
    """
    核心订单撮合算法 - 计算收盘价和成交明细

    该函数实现了完整的股票订单撮合逻辑，模拟真实交易所的撮合机制。
    采用价格优先、时间优先的原则，寻找最大成交量的价格点作为收盘价。

    撮合规则：
    1. 价格优先：买方出价高者优先，卖方要价低者优先
    2. 时间优先：同价格下，先下单者优先
    3. 最大成交量原则：选择能产生最大成交量的价格作为成交价
    4. 涨跌停限制：成交价必须在上日收盘价±10%范围内
    5. 时间戳去重：自动处理重复时间戳问题

    算法流程：
    1. 订单排序和时间戳去重
    2. 检查是否存在可撮合的订单
    3. 构建价格-数量映射表
    4. 遍历所有价格点寻找最大成交量
    5. 生成详细的成交记录
    6. 创建订单簿可视化

    Args:
        buy_orders (list[Order]): 买入订单列表
        sell_orders (list[Order]): 卖出订单列表
        last_price (float): 上一交易日收盘价
        current_date (str, optional): 当前交易日期，用于可视化
        output_dir (str, optional): 输出目录，用于保存可视化文件

    Returns:
        tuple[float, int, list[dict]]: 包含以下元素的元组
            - 成交价格 (float): 最终确定的收盘价
            - 总成交量 (int): 总的股票成交数量
            - 成交记录列表 (list[dict]): 每笔成交的详细信息

    Note:
        - 如果没有买卖双方订单，返回上一日收盘价
        - 如果买卖价格无法撮合，返回上一日收盘价
        - 会自动生成订单簿可视化图表
        - 成交记录包含完整的用户、价格、数量信息
    """
    # 如果没有订单,返回上一个交易日收盘价
    if not buy_orders or not sell_orders:
        return last_price, 0, []

    # 按价格和时间排序
    buy_orders = sorted(buy_orders, key=lambda x: (-x.price, x.timestamp))
    sell_orders = sorted(sell_orders, key=lambda x: (x.price, x.timestamp))

    # 检查并修复重复时间戳
    def fix_duplicate_timestamps(orders: list[Order]) -> None:
        """修复重复的时间戳"""
        seen_timestamps = set()
        for order in orders:
            while order.timestamp in seen_timestamps:
                # 如果时间戳重复，添加1毫秒
                order.timestamp += timedelta(microseconds=1000)
            seen_timestamps.add(order.timestamp)

    # 处理买卖单中的重复时间戳
    fix_duplicate_timestamps(buy_orders)
    fix_duplicate_timestamps(sell_orders)

    # 计算涨跌停限制
    upper_limit = last_price * 1.1
    lower_limit = last_price * 0.9

    # 可以添加调试信息
    # if len(buy_orders) > 1:
    #     for i in range(len(buy_orders)-1):
    #         if buy_orders[i].price == buy_orders[i+1].price:
    #             print(f"买单时时间优先验证: {buy_orders[i].timestamp} 早于 {buy_orders[i+1].timestamp}")

    # if len(sell_orders) > 1:
    #     for i in range(len(sell_orders)-1):
    #         if sell_orders[i].price == sell_orders[i+1].price:
    #             print(f"卖单时间优先验证: {sell_orders[i].timestamp} 早于 {sell_orders[i+1].timestamp}")

    # 打印排序后的订单，验证时间戳是否生效
    # print("\n买单排序:")
    # for order in buy_orders:
    #     print(f"价格: {order.price}, 数量: {order.quantity}, "
    #           f"时间: {order.timestamp.strftime('%H:%M:%S.%f')}")

    # print("\n卖单排序:")
    # for order in sell_orders:
    #     print(f"价格: {order.price}, 数量: {order.quantity}, "
    #           f"时间: {order.timestamp.strftime('%H:%M:%S.%f')}")

    # 检查是否有可能的成交
    if buy_orders[0].price < sell_orders[0].price:
        return last_price, 0, []

    # 整理买单和卖单：价格到数量的映射
    buy_map = defaultdict(int)
    sell_map = defaultdict(int)

    for order in buy_orders:
        buy_map[order.price] += order.quantity
    for order in sell_orders:
        sell_map[order.price] += order.quantity

    # 收集所有可能的价格点
    price_points = sorted(set(buy_map.keys()).union(sell_map.keys()))

    # 添加调试信息
    # print("\n价格映射:")
    # print("买单价格映射:")
    # for price, quantity in buy_map.items():
    #     print(f"价格: {price:.2f}, 总数量: {quantity}")
    # print("\n卖单价格映射:")
    # for price, quantity in sell_map.items():
    #     print(f"价格: {price:.2f}, 总数量: {quantity}")

    # print("\n所有价格点:", [f"{p:.2f}" for p in price_points])

    # 继续撮合过程
    max_volume = 0
    best_price = last_price

    # 遍历每个价格点
    for price in price_points:
        # print(f"\n尝试价格点: {price:.2f}")
        buy_volume = sum(buy_map[p] for p in buy_map if p >= price)
        sell_volume = sum(sell_map[p] for p in sell_map if p <= price)
        volume = min(buy_volume, sell_volume)
        # print(f"该价格点的买量: {buy_volume}, 卖量: {sell_volume}, 可成交量: {volume}")

        if volume > max_volume:
            max_volume = volume
            best_price = price
            # print(f"更新最佳价格: {best_price:.2f}, 最大成交量: {max_volume}")

    if best_price is None:
        return last_price, 0, []

    # 计算每个订单的实际成交情况
    transactions = []
    remaining_volume = max_volume

    # 处理买单成交
    for order in buy_orders:
        if order.price >= best_price and remaining_volume > 0:
            executed_quantity = min(order.quantity, remaining_volume)
            if executed_quantity > 0:
                transactions.append(
                    {
                        "stock_code": order.stock_code,
                        "user_id": order.user_id,
                        "direction": "buy",
                        "executed_price": best_price,
                        "executed_quantity": executed_quantity,
                        "original_quantity": order.quantity,
                        "unfilled_quantity": order.quantity - executed_quantity,
                        "timestamp": order.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    }
                )
                remaining_volume -= executed_quantity

    # 处理卖单成交
    remaining_volume = max_volume
    for order in sell_orders:
        if order.price <= best_price and remaining_volume > 0:
            executed_quantity = min(abs(order.quantity), remaining_volume)
            if executed_quantity > 0:
                transactions.append(
                    {
                        "stock_code": order.stock_code,
                        "user_id": order.user_id,
                        "direction": "sell",
                        "executed_price": best_price,
                        "executed_quantity": executed_quantity,
                        "original_quantity": order.quantity,
                        "unfilled_quantity": order.quantity - executed_quantity,
                        "timestamp": order.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    }
                )
                remaining_volume -= executed_quantity

    # 在返回结果前生成可视化
    if best_price and current_date:
        try:
            # print(f"\n正在生成订单簿可视化 - {current_date}")
            visualize_order_book(
                buy_orders, sell_orders, best_price, current_date, output_dir
            )
            # print(f"可视化已保存到 simulation_results/order_book_{current_date}.png")
        except Exception as e:
            print(f"生成订单簿可视化时出错: {str(e)}")

    return best_price, max_volume, transactions


def visualize_order_book(
    buy_orders: list[Order],
    sell_orders: list[Order],
    closing_price: float,
    date: str,
    save_path: str = "simulation_results",
):
    """Visualize order book with text format and save order data"""
    # 计算买卖单占比
    total_orders = len(buy_orders) + len(sell_orders)
    buy_ratio = len(buy_orders) / total_orders * 100
    sell_ratio = len(sell_orders) / total_orders * 100

    # 统计订单数据
    buy_prices = sorted(
        set(order.price for order in buy_orders), reverse=True
    )  # 买单从高到低
    sell_prices = sorted(set(order.price for order in sell_orders))  # 卖单从低到高

    buy_price_map = defaultdict(lambda: {"total_quantity": 0, "count": 0})
    for order in buy_orders:
        buy_price_map[order.price]["total_quantity"] += order.quantity
        buy_price_map[order.price]["count"] += 1

    sell_price_map = defaultdict(lambda: {"total_quantity": 0, "count": 0})
    for order in sell_orders:
        sell_price_map[order.price]["total_quantity"] += order.quantity
        sell_price_map[order.price]["count"] += 1

    # 生成文本可视化
    visualization = []
    visualization.append(f"Order Book - {date}")
    visualization.append(f"Buy {buy_ratio:.2f}% | Sell {sell_ratio:.2f}%")
    visualization.append(f"Closing Price: {closing_price:.2f}")
    visualization.append("")

    # 计算最大宽度
    max_buy_width = (
        max(
            [
                len(
                    f"{i+1:2d} {price:.2f} {buy_price_map[price]['total_quantity']:4d} ({buy_price_map[price]['count']})"
                )
                for i, price in enumerate(buy_prices)
            ]
        )
        if buy_prices
        else 0
    )
    max_sell_width = (
        max(
            [
                len(
                    f"{i+1:2d} {price:.2f} {sell_price_map[price]['total_quantity']:4d} ({sell_price_map[price]['count']})"
                )
                for i, price in enumerate(sell_prices)
            ]
        )
        if sell_prices
        else 0
    )

    # 表头
    visualization.append(
        "Buy Orders".ljust(max_buy_width) + " | " + "Sell Orders".ljust(max_sell_width)
    )
    visualization.append("-" * (max_buy_width + 3 + max_sell_width))

    # 合并买卖单显示
    max_rows = max(len(buy_prices), len(sell_prices))
    for i in range(max_rows):
        buy_str = ""
        sell_str = ""

        if i < len(buy_prices):
            price = buy_prices[i]
            info = buy_price_map[price]
            buy_str = (
                f"{i+1:2d} {price:.2f} {info['total_quantity']:4d} ({info['count']})"
            )
            if price >= closing_price:
                buy_str += " <-- Close"

        if i < len(sell_prices):
            price = sell_prices[i]
            info = sell_price_map[price]
            sell_str = (
                f"{i+1:2d} {price:.2f} {info['total_quantity']:4d} ({info['count']})"
            )
            if price >= closing_price and (
                i == 0 or sell_prices[i - 1] < closing_price
            ):
                sell_str += " <-- Close"

        visualization.append(
            f"{buy_str.ljust(max_buy_width)} | {sell_str.ljust(max_sell_width)}"
        )

    # 获取股票代码
    stock_code = (
        buy_orders[0].stock_code
        if buy_orders
        else sell_orders[0].stock_code if sell_orders else "unknown"
    )

    # 确保目录存在
    order_book_dir = f"{save_path}/order_books/{stock_code}"
    os.makedirs(order_book_dir, exist_ok=True)

    # 保存文本可视化到与json相同的目录
    with open(f"{order_book_dir}/{date}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(visualization))

    # 保存原始数据为JSON
    order_data = {
        "date": date,
        "stock_code": stock_code,
        "closing_price": closing_price,
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "buy_orders": [
            {
                "price": price,
                "quantity": buy_price_map[price]["total_quantity"],
                "count": buy_price_map[price]["count"],
            }
            for price in buy_prices
        ],
        "sell_orders": [
            {
                "price": price,
                "quantity": sell_price_map[price]["total_quantity"],
                "count": sell_price_map[price]["count"],
            }
            for price in sell_prices
        ],
    }

    # 保存JSON数据
    with open(f"{order_book_dir}/{date}.json", "w", encoding="utf-8") as f:
        json.dump(order_data, f, indent=2, ensure_ascii=False)


def process_daily_orders(
    orders: list[Order],
    last_prices: dict,
    current_date: str,
    output_dir: str,
    json_file_path: str,
) -> dict:
    """
    处理所有股票的每日订单

    Args:
        orders: 所有订单列表 [order1, order2]
        last_prices: 字典，键为股票代码，值为上一交易日收盘价
        current_date: 当前交易日期

    Returns:
        dict: 每支股票的交易结果，格式为：
        {
            'stock_code': {
                'closing_price': float,
                'volume': int,
                'transactions': list[dict]
            }
        }
    """
    # 按股票代码分组订单
    stock_orders = defaultdict(lambda: {"buy": [], "sell": []})
    for order in orders:
        if order.direction == "buy":
            stock_orders[order.stock_code]["buy"].append(order)
        else:  # sell
            stock_orders[order.stock_code]["sell"].append(order)

    for stock_code, orders_dict in stock_orders.items():

        for order in orders_dict["buy"]:
            order.user_id = order.user_id.split("_")[0]

        for order in orders_dict["sell"]:
            order.user_id = order.user_id.split("_")[0]

    print("\n=== 复制订单统计 ===")

    # 处理订单复制
    for stock_code, orders_dict in stock_orders.items():
        buy_orders = orders_dict["buy"]
        sell_orders = orders_dict["sell"]

        # 如果买卖双方都有订单才进行分析
        if buy_orders and sell_orders:
            buy_total_quantity = sum(order.quantity for order in buy_orders)
            sell_total_quantity = sum(order.quantity for order in sell_orders)

            # 计算比率并向下取整
            if buy_total_quantity > sell_total_quantity:
                ratio = buy_total_quantity / sell_total_quantity
                int_ratio = buy_total_quantity // sell_total_quantity
                if ratio >= 2.5:
                    # 获取该股票现有的时间戳
                    existing_timestamps = {
                        order.timestamp for order in buy_orders + sell_orders
                    }
                    # 复制卖单
                    new_sell_orders = []
                    copies = min(int_ratio - 1, 3)  # 最多复制3份

                    for _ in range(copies):
                        for original_order in sell_orders:
                            # 生成新的不重复时间戳
                            new_timestamp = generate_unique_timestamp(
                                current_date, existing_timestamps
                            )
                            existing_timestamps.add(new_timestamp)

                            new_order = Order(
                                stock_code=original_order.stock_code,
                                price=original_order.price,
                                quantity=original_order.quantity,
                                timestamp=new_timestamp,
                                user_id="ZYF",
                                direction="sell",
                            )
                            new_sell_orders.append(new_order)

                    orders_dict["sell"].extend(new_sell_orders)
                    print(f"{stock_code}：复制 {copies} 份卖单")

            elif sell_total_quantity > buy_total_quantity:
                ratio = sell_total_quantity / buy_total_quantity
                int_ratio = sell_total_quantity // buy_total_quantity
                if ratio >= 2.5:
                    # 获取该股票现有的时间戳
                    existing_timestamps = {
                        order.timestamp for order in buy_orders + sell_orders
                    }
                    # 复制买单
                    new_buy_orders = []
                    copies = min(int_ratio - 1, 3)  # 最多复制3份

                    for _ in range(copies):
                        for original_order in buy_orders:
                            # 生成新的不重复时间戳
                            new_timestamp = generate_unique_timestamp(
                                current_date, existing_timestamps
                            )
                            existing_timestamps.add(new_timestamp)

                            new_order = Order(
                                stock_code=original_order.stock_code,
                                price=original_order.price,
                                quantity=original_order.quantity,
                                timestamp=new_timestamp,
                                user_id="ZYF",
                                direction="buy",
                            )
                            new_buy_orders.append(new_order)

                    orders_dict["buy"].extend(new_buy_orders)
                    print(f"{stock_code}：复制 {copies} 份买单")

    # 保存stock_orders到json文件
    orders_data = {}
    for stock_code, orders_dict in stock_orders.items():
        orders_data[stock_code] = {
            "buy": [
                {
                    "price": order.price,
                    "quantity": order.quantity,
                    "timestamp": order.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "user_id": order.user_id,
                }
                for order in orders_dict["buy"]
            ],
            "sell": [
                {
                    "price": order.price,
                    "quantity": order.quantity,
                    "timestamp": order.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "user_id": order.user_id,
                }
                for order in orders_dict["sell"]
            ],
        }

    # 构建新的json文件路径
    orders_file_path = json_file_path.replace(".json", "_orders.json")

    # 保存orders数据
    with open(orders_file_path, "w", encoding="utf-8") as f:
        json.dump(orders_data, f, indent=2, ensure_ascii=False)

    print(f"订单数据已保存到 {orders_file_path}")
    print("==================")

    # 处理每支股票的订单
    results = {}
    for stock_code, stock_order in stock_orders.items():
        # print(f"\n处理股票 {stock_code} 的撮合...")
        closing_price, volume, transactions = calculate_closing_price(
            stock_order["buy"],
            stock_order["sell"],
            last_prices.get(stock_code, 0),
            current_date,
            output_dir,
        )

        # print(f"撮合结果: 收盘价={closing_price:.2f}, 成交量={volume}")
        # if transactions:
        #     print("成交明细:")
        #     for trans in transactions:
        #         print(f"方向: {trans['direction']}, 价格: {trans['executed_price']:.2f}, "
        #               f"数量: {trans['executed_quantity']}")

        results[stock_code] = {
            "closing_price": closing_price,
            "volume": volume,
            "transactions": transactions,
        }

    return results


def generate_unique_timestamp(current_date: str, existing_timestamps: set) -> datetime:
    morning_start = datetime.strptime(f"{current_date} 09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end = datetime.strptime(f"{current_date} 11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start = datetime.strptime(f"{current_date} 13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end = datetime.strptime(f"{current_date} 15:00:00", "%Y-%m-%d %H:%M:%S")

    morning_seconds = (morning_end - morning_start).total_seconds()
    afternoon_seconds = (afternoon_end - afternoon_start).total_seconds()
    total_seconds = morning_seconds + afternoon_seconds

    while True:
        random_seconds = random.uniform(0, total_seconds)
        if random_seconds < morning_seconds:
            timestamp = morning_start + timedelta(seconds=random_seconds)
        else:
            timestamp = afternoon_start + timedelta(
                seconds=random_seconds - morning_seconds
            )

        if timestamp not in existing_timestamps:
            return timestamp


from collections import defaultdict
import pandas as pd


def save_daily_results(
    results: dict, date: str, output_dir: str = "simulation_results"
):
    """
    保存每日交易结果，并计算大单资金流向

    Args:
        results: 交易结果字典
        date: 交易日期
        output_dir: 输出目录
    """
    # 统计变量初始化
    total_transactions = 0
    total_volume = 0
    total_amount = 0

    # 保存交易汇总
    summary = []
    large_order_flow = defaultdict(float)  # 用于统计大单资金流向
    user_stock_accumulated = defaultdict(float)  # 用于记录用户对每只股票的累计交易金额
    user_stock_transactions = defaultdict(list)  # 用于记录用户对每只股票的所有交易

    # 第一步：遍历所有交易，计算累计金额并记录交易
    for stock_code, result in results.items():
        for trans in result["transactions"]:
            # 计算交易金额
            transaction_amount = trans["executed_price"] * trans["executed_quantity"]
            total_amount += transaction_amount
            total_volume += trans["executed_quantity"]

            # 更新用户对股票的累计交易金额
            user_stock_key = (trans["user_id"], stock_code)
            user_stock_accumulated[user_stock_key] += transaction_amount
            user_stock_transactions[user_stock_key].append(trans)

        total_transactions += len(result["transactions"])

    # 第二步：统一处理大单逻辑
    large_order_count = 0
    for stock_code, result in results.items():
        net_inflow = 0
        for user_stock_key, transactions in user_stock_transactions.items():
            if user_stock_key[1] == stock_code:  # 只处理当前股票
                # 判断是否为大单
                if (
                    user_stock_accumulated[user_stock_key] >= 1000000
                ):  # 累计金额 ≥ 100 万
                    large_order_count += 1
                    # 计算资金流向
                    for trans in transactions:
                        if trans["direction"] == "buy":
                            net_inflow += (
                                trans["executed_price"] * trans["executed_quantity"]
                            )
                        else:  # sell
                            net_inflow -= (
                                trans["executed_price"] * trans["executed_quantity"]
                            )

        large_order_flow[stock_code] = net_inflow

        summary.append(
            {
                "date": date,
                "stock_code": stock_code,
                "closing_price": result["closing_price"],
                "volume": result["volume"],
                "transaction_count": len(result["transactions"]),
                "large_order_net_inflow": net_inflow,
            }
        )

    # 打印交易统计信息
    print("\n=== 成交统计 ===")
    print(f"总成交笔数: {total_transactions}")
    print(f"总成交量: {total_volume:,}")
    print(f"总成交金额: {total_amount/10000:,.2f}万")
    print(f"大单数量: {large_order_count}")
    print("==================\n")

    # 保存为CSV文件
    df = pd.DataFrame(summary)
    df.to_csv(f"{output_dir}/daily_summary_{date}.csv", index=False)

    # 保存详细交易记录
    transactions = []
    for stock_code, result in results.items():
        transactions.extend(result["transactions"])

    # 定义交易记录的列
    columns = [
        "stock_code",
        "user_id",
        "direction",
        "executed_price",
        "executed_quantity",
        "original_quantity",
        "unfilled_quantity",
        "timestamp",
    ]

    # 创建DataFrame，如果transactions为空，将创建一个只有列名的空DataFrame
    df_trans = pd.DataFrame(transactions if transactions else [], columns=columns)
    df_trans.to_csv(f"{output_dir}/transactions_{date}.csv", index=False)

    # 单独保存大单资金流向数据
    large_order_data = [
        {"date": date, "stock_code": stock_code, "large_order_net_inflow": net_inflow}
        for stock_code, net_inflow in large_order_flow.items()
    ]

    df_large_order = pd.DataFrame(large_order_data)
    df_large_order.to_csv(f"{output_dir}/large_order_flow_{date}.csv", index=False)


def create_orders_from_decisions(
    decisions: list[dict], current_date: str
) -> list[Order]:
    """
    将用户交易决策转换为标准订单对象

    该函数将交易代理生成的抽象决策转换为具体的交易订单，
    并为每个订单分配随机的交易时间戳，模拟真实的交易时间分布。

    处理流程：
    1. 定义交易时间段（上午9:30-11:30，下午13:00-15:00）
    2. 为每个有效决策生成随机时间戳
    3. 确保时间戳的唯一性
    4. 创建标准Order对象
    5. 按时间戳排序返回

    Args:
        decisions (list[dict]): 用户交易决策列表，每个决策包含：
            - user_id: 用户ID
            - stock_code: 股票代码
            - direction: 交易方向（'buy'/'sell'）
            - amount: 交易数量
            - target_price: 目标价格
        current_date (str): 当前交易日期，格式'YYYY-MM-DD'

    Returns:
        list[Order]: 按时间戳排序的订单对象列表

    Note:
        - 只处理有效的买卖决策（数量>0）
        - 时间戳在交易时间段内随机分布
        - 确保所有时间戳的唯一性
        - 所有订单数量都使用正数表示
    """
    # 定义交易时间段
    morning_start = datetime.strptime(f"{current_date} 09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end = datetime.strptime(f"{current_date} 11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start = datetime.strptime(f"{current_date} 13:00:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end = datetime.strptime(f"{current_date} 15:00:00", "%Y-%m-%d %H:%M:%S")

    # 计算总可用秒数
    morning_seconds = (morning_end - morning_start).total_seconds()
    afternoon_seconds = (afternoon_end - afternoon_start).total_seconds()
    total_seconds = morning_seconds + afternoon_seconds

    # 生成不重复的随机时间戳
    used_timestamps = set()
    orders = []

    for decision in decisions:
        if decision["direction"] in ["buy", "sell"] and decision["amount"] > 0:
            while True:
                # 随机选择一个时间点
                random_seconds = random.uniform(0, total_seconds)

                if random_seconds < morning_seconds:
                    # 上午交易时段
                    timestamp = morning_start + timedelta(seconds=random_seconds)
                else:
                    # 下午交易时段
                    timestamp = afternoon_start + timedelta(
                        seconds=random_seconds - morning_seconds
                    )

                # 确保时间戳唯一
                if timestamp not in used_timestamps:
                    used_timestamps.add(timestamp)
                    break

            order = Order(
                stock_code=decision["stock_code"],
                price=decision["target_price"],
                quantity=decision["amount"],  # 所有数量都为正数
                timestamp=timestamp,
                user_id=decision["user_id"],
                direction=decision["direction"],
            )
            orders.append(order)

    # 按时间戳排序
    orders.sort(key=lambda x: x.timestamp)
    return orders


def process_trading_day(
    decisions: list[dict],
    last_prices: dict,
    current_date: str,
    output_dir: str = "simulation_results",
    db_path: str = "data/sys_100.db",
    df_stock: pd.DataFrame = None,
    df_stock_profile_real: pd.DataFrame = None,
    json_file_path: str = None,
):
    """
    处理单个交易日的完整交易流程

    该函数是交易日处理的核心协调器，负责将用户决策转换为订单，
    执行撮合交易，并更新所有相关的数据库表。

    完整处理流程：
    1. 决策转换：将用户交易决策转换为标准订单格式
    2. 时间戳验证：确保所有订单时间戳的唯一性
    3. 订单撮合：执行多股票并行撮合处理
    4. 结果保存：保存交易结果到CSV文件
    5. 数据库更新：更新股票数据表
    6. 交易明细：更新交易明细表
    7. 用户档案：更新用户持仓和收益信息

    Args:
        decisions (list[dict]): 用户交易决策列表
        last_prices (dict): 上一交易日各股票收盘价，格式{stock_code: price}
        current_date (str): 当前交易日期，格式'YYYY-MM-DD'
        output_dir (str): 结果输出目录，默认"simulation_results"
        db_path (str): 数据库文件路径
        df_stock (pd.DataFrame): 股票历史数据
        df_stock_profile_real (pd.DataFrame): 真实股票资料数据
        json_file_path (str): 原始决策JSON文件路径

    Returns:
        dict: 每支股票的交易结果，包含收盘价、成交量、交易明细等

    Note:
        - 支持多股票并行处理
        - 包含完整的数据验证和异常处理
        - 会自动更新所有相关数据库表
    """
    # 1. 转换决策为订单，并分配随机时间戳
    orders = create_orders_from_decisions(decisions, current_date)

    # 2. 验证时间戳是否有重复
    assert validate_order_timestamps(orders), "存在重复时间戳"

    # 3. 处理订单
    results = process_daily_orders(
        orders, last_prices, current_date, output_dir, json_file_path
    )

    # 4. 保存结果
    save_daily_results(results, current_date, output_dir)

    # 5. 更新数据库中的StockData表
    update_stock_data_table(
        results=results,
        current_date=current_date,
        output_dir=output_dir,
        db_path=db_path,
        df_stock=df_stock,
    )

    # 更新Tradingdetails表
    update_trading_details_table(
        current_date, db_path, output_dir, df_stock_profile_real
    )

    # 更新Profiles表
    update_profiles_table(current_date, db_path, output_dir)

    return results


def update_stock_data_table(
    results: dict,
    current_date: str,
    db_path: str = "data/sys_100.db",
    real_data_path: str = "data/stock_data.csv",
    output_dir: str = "simulation_results",
    df_stock: pd.DataFrame = None,
):
    """
    更新数据库中的股票数据表（StockData）

    该函数负责将当日的交易结果更新到数据库的StockData表中，包括：
    - 基础价格数据（收盘价、涨跌幅等）
    - 技术指标（移动平均线、MACD等）
    - 估值指标（市盈率、市净率等）
    - 成交量数据和资金流向

    数据更新策略：
    1. 使用撮合结果更新有交易的股票数据
    2. 使用真实市场数据更新无交易的股票数据
    3. 基于真实数据等比例调整估值指标
    4. 计算技术指标的移动平均值

    Args:
        results (dict): 撮合交易结果字典，包含每支股票的收盘价、成交量等
        current_date (str): 当前交易日期，格式为'YYYY-MM-DD'
        db_path (str): SQLite数据库文件路径
        real_data_path (str): 真实历史数据CSV文件路径
        output_dir (str): 输出目录路径
        df_stock (pd.DataFrame): 股票历史数据DataFrame

    Raises:
        FileNotFoundError: 当必要的数据文件不存在时抛出
        Exception: 数据库操作失败时抛出

    Note:
        - 估值指标基于真实数据和模拟价格的比例进行调整
        - 技术指标计算支持不足周期的情况
        - 会自动处理有交易和无交易的股票
    """
    try:
        # 读取大单资金流数据（模拟的）
        large_order_file = f"{output_dir}/large_order_flow_{current_date}.csv"
        if not os.path.exists(large_order_file):
            raise FileNotFoundError(f"大单资金流文件 {large_order_file} 不存在")
        large_order_df = pd.read_csv(large_order_file)

        # 检查DataFrame是否为空
        if not large_order_df.empty:
            large_order_df["date"] = pd.to_datetime(large_order_df["date"])
        else:
            # 如果DataFrame为空，创建一个空的DataFrame但包含所需的列
            large_order_df = pd.DataFrame(
                columns=["date", "stock_code", "large_order_net_inflow"]
            )
            large_order_df["date"] = large_order_df["date"].astype("datetime64[ns]")

        current_date = pd.to_datetime(current_date)

        # 读取真实的历史数据（包含到2025年的数据）
        if not os.path.exists(real_data_path):
            raise FileNotFoundError(f"历史交易数据文件 {real_data_path} 不存在")
        historical_data_real = pd.read_csv(real_data_path)
        historical_data_real["date"] = pd.to_datetime(historical_data_real["date"])

        # 连接数据库
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 获取所有股票代码--全部上证50
            all_stock_codes = df_stock["stock_id"].unique()

            # 更新每支股票的数据
            for stock_code in all_stock_codes:
                # 获取当日的真实交易数据作为基准
                current_real_data = (
                    historical_data_real[
                        (historical_data_real["ts_code"] == stock_code)
                        & (historical_data_real["date"] <= current_date)
                    ]
                    .sort_values("date", ascending=False)
                    .iloc[0]
                )

                # 获取该股票交易数据（模拟）
                stock_history_sim = df_stock[
                    df_stock["stock_id"] == stock_code
                ].sort_values("date", ascending=False)
                prev_data = stock_history_sim[
                    stock_history_sim["date"] < current_date
                ].iloc[0]
                prev_close = prev_data["close_price"]

                if stock_code in results:
                    result = results[stock_code]
                    closing_price = result["closing_price"]
                    volume = result["volume"]

                    # 检查是否存在匹配的记录
                    if not large_order_df[
                        large_order_df["stock_code"] == stock_code
                    ].empty:
                        large_order_net = large_order_df[
                            large_order_df["stock_code"] == stock_code
                        ]["large_order_net_inflow"].iloc[0]
                    else:
                        large_order_net = 0

                    # 计算股价变动和涨跌幅
                    price_change = closing_price - prev_close
                    pct_change = (price_change / prev_close) * 100

                else:
                    real_stock_data = current_real_data
                    if not real_stock_data.empty:
                        # 使用实际涨跌幅更新价格
                        pct_change = real_stock_data["pct_chg"]
                        closing_price = prev_close * (1 + pct_change / 100)
                        price_change = closing_price - prev_close
                    else:
                        # 如果没有实际数据，保持价格不变
                        closing_price = prev_data["close_price"]
                        price_change = 0
                        pct_change = 0

                    volume = 10000  # 设置默认交易量
                    large_order_net = 0  # 没有大单资金流入

                # 使用当日真实数据作为基准，根据模拟收盘价等比例计算估值指标
                price_ratio = closing_price / current_real_data["close"]
                new_pe_ttm = current_real_data["pe_ttm"] * price_ratio
                new_pb = current_real_data["pb"] * price_ratio
                new_ps_ttm = current_real_data["ps_ttm"] * price_ratio
                new_dv_ttm = current_real_data["dv_ttm"] / price_ratio

                # 添加最新的收盘价和成交量
                new_row = pd.DataFrame(
                    {
                        "stock_id": [stock_code],
                        "close_price": [closing_price],
                        "pre_close": [prev_close],
                        "change": [price_change],
                        "pct_chg": [pct_change],
                        "pe_ttm": [new_pe_ttm],
                        "pb": [new_pb],
                        "ps_ttm": [new_ps_ttm],
                        "dv_ttm": [new_dv_ttm],
                        "vol": [volume],
                        "vol_5": [None],
                        "vol_10": [None],
                        "vol_30": [None],
                        "ma_hfq_5": [None],
                        "ma_hfq_10": [None],
                        "ma_hfq_30": [None],
                        "elg_amount_net": [large_order_net],
                        "date": [pd.to_datetime(current_date)],
                    }
                )
                stock_history_sim = pd.concat(
                    [stock_history_sim, new_row], ignore_index=True
                )
                stock_history_sim = stock_history_sim.sort_values(
                    "date", ascending=False
                )

                # 计算技术指标
                vol5 = (
                    stock_history_sim["vol"]
                    .iloc[::-1]
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                vol10 = (
                    stock_history_sim["vol"]
                    .iloc[::-1]
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                vol30 = (
                    stock_history_sim["vol"]
                    .iloc[::-1]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .iloc[-1]
                )

                ma5 = (
                    stock_history_sim["close_price"]
                    .iloc[::-1]
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ma10 = (
                    stock_history_sim["close_price"]
                    .iloc[::-1]
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ma30 = (
                    stock_history_sim["close_price"]
                    .iloc[::-1]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .iloc[-1]
                )

                if isinstance(current_date, pd.Timestamp):
                    current_date = current_date.strftime("%Y-%m-%d")
                try:
                    cursor.execute(
                        """
                    INSERT INTO StockData (
                        stock_id,
                        date,
                        close_price,
                        pre_close,
                        change,
                        pct_chg,
                        pe_ttm,
                        pb,
                        ps_ttm,
                        dv_ttm,
                        vol,
                        vol_5,
                        vol_10,
                        vol_30,
                        elg_amount_net,
                        ma_hfq_5,
                        ma_hfq_10,
                        ma_hfq_30
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            stock_code,
                            current_date,
                            round(float(closing_price), 2),
                            round(float(prev_close), 2),
                            round(float(price_change), 2),
                            round(float(pct_change), 4),
                            round(float(new_pe_ttm), 4),
                            round(float(new_pb), 4),
                            round(float(new_ps_ttm), 4),
                            round(float(new_dv_ttm), 4),
                            int(volume) if pd.notna(volume) else None,
                            round(float(vol5), 2) if pd.notna(vol5) else None,
                            round(float(vol10), 2) if pd.notna(vol10) else None,
                            round(float(vol30), 2) if pd.notna(vol30) else None,
                            round(float(large_order_net), 2),
                            round(float(ma5), 5),
                            round(float(ma10), 5),
                            round(float(ma30), 5),
                        ),
                    )
                except Exception as e:
                    print(f"插入数据时发生错误: {e}")

                # 提交更改
                conn.commit()

    except Exception as e:
        # print(f"更新股票数据时发生错误: {e}")
        raise


def update_trading_details_table(
    current_date: str,
    db_path: str = "data/sys_100.db",
    output_dir: str = "simulation_results",
    df_stock_profile_real: pd.DataFrame = None,
):
    """
    更新数据库中的 TradingDetails 表。

    Args:
        current_date (str): 当前交易日期，格式为 'YYYY-MM-DD'。
        db_path (str): 数据库文件路径，默认为 'sys_100.db'。
    """
    try:
        transaction_file_path = f"{output_dir}/transactions_{current_date}.csv"
        if not os.path.exists(transaction_file_path):
            # print(f"交易文件 {transaction_file_path} 不存在")
            return

        transaction_df = pd.read_csv(transaction_file_path)
        # 如果交易记录为空，直接返回
        if transaction_df.empty:
            return

        # 保留到日频
        transaction_df["timestamp"] = pd.to_datetime(transaction_df["timestamp"])
        transaction_df["timestamp"] = transaction_df["timestamp"].dt.strftime(
            "%Y-%m-%d"
        )

        # 合并数据
        merged_df = pd.merge(
            transaction_df,
            df_stock_profile_real,
            left_on="stock_code",
            right_on="stock_id",
            how="left",
        )

        # TODO
        merged_df = merged_df[merged_df["user_id"] != "ZYF"]
        merged_df = merged_df.reset_index(drop=True)

        # 连接数据库
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 遍历 DataFrame 的每一行
            for _, row in merged_df.iterrows():
                # 插入交易详情到 TradingDetails 表
                cursor.execute(
                    """
                INSERT INTO TradingDetails (
                    user_id,
                    date_time,
                    industry,
                    stock_id,
                    price,
                    stock_name,
                    trading_direction,
                    volume,
                    valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        row["user_id"],
                        row["timestamp"],
                        row["industry"],
                        row["stock_code"],
                        round(float(row["executed_price"]), 2),
                        row["name"],
                        row["direction"],
                        int(row["executed_quantity"]),
                        True,
                    ),
                )

            # 提交更改
            conn.commit()

    except Exception as e:
        # print(f"更新 TradingDetails 表时发生错误: {e}")
        raise


def update_profiles_table(current_date: str, db_path: str, output_dir: str):
    """
    根据交易结果更新用户档案表

    Args:
        current_date: 当前交易日期
        db_path: 数据库路径
        output_dir: 输出目录
    """

    try:
        # 读取交易数据
        # 确保都是当前有的人  ZYF
        transaction_file = f"{output_dir}/transactions_{current_date}.csv"
        if not os.path.exists(transaction_file):
            raise FileNotFoundError(f"交易文件 {transaction_file} 不存在")

        transactions_df = pd.read_csv(transaction_file)

        # 确保user_id类型一致，且不包含ZYF
        transactions_df["user_id"] = transactions_df["user_id"].astype(str)
        # transactions_df=transactions_df[transactions_df['user_id']!='ZYF']
        # transactions_df=transactions_df.reset_index(drop=True)

        current_date_obj = pd.to_datetime(current_date)
        previous_date = (current_date_obj - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d 00:00:00"
        )

        # 连接数据库并获取所有用户信息
        with sqlite3.connect(db_path) as conn:

            # 获取新交易的用户
            user_industries = {}
            df_trading_details = pd.read_sql_query(
                f"""SELECT * FROM TradingDetails where date_time = ?""",
                conn,
                params=(current_date,),
            )
            new_users_list = list(df_trading_details["user_id"].unique())

            cursor = conn.cursor()

            # 获取所有用户信息
            cursor.execute(
                "SELECT * FROM Profiles where created_at = ?", (previous_date,)
            )
            profiles = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            # 处理每个用户的更新
            for profile in profiles:
                profile_dict = dict(zip(columns, profile))
                user_id = str(profile_dict["user_id"])

                if user_id in new_users_list:
                    query = """
                        SELECT industry, COUNT(industry) AS count
                        FROM TradingDetails
                        WHERE user_id = ? AND trading_direction = 'buy'
                        GROUP BY industry
                        ORDER BY count DESC
                        LIMIT 5
                    """
                    cursor.execute(query, (user_id,))
                    industry_list = cursor.fetchall()
                    profile_dict["fol_ind"] = json.dumps(
                        list(dict(industry_list).keys()), ensure_ascii=False
                    )

                # 获取用户的交易记录
                user_trades = transactions_df[transactions_df["user_id"] == user_id]
                # 读取现有状态
                current_cash = float(profile_dict["current_cash"])
                cur_positions = json.loads(profile_dict["cur_positions"])
                ini_cash = float(profile_dict["ini_cash"])

                # 处理每笔交易
                for _, trade in user_trades.iterrows():
                    stock_id = trade["stock_code"]
                    price = float(trade["executed_price"])
                    quantity = int(trade["executed_quantity"])
                    direction = trade["direction"]

                    if direction == "buy":
                        current_cash -= price * quantity
                        if stock_id not in cur_positions:
                            cur_positions[stock_id] = {"shares": 0, "ratio": 0.0}
                        cur_positions[stock_id]["shares"] += quantity
                    else:  # sell
                        current_cash += price * quantity
                        cur_positions[stock_id]["shares"] -= quantity
                        if cur_positions[stock_id]["shares"] <= 0:
                            del cur_positions[stock_id]

                # 获取最新股价
                stock_prices = {}
                cursor.execute(
                    "SELECT stock_id, close_price FROM StockData WHERE date = ?",
                    (current_date,),
                )
                rows = cursor.fetchall()
                stock_prices = {row[0]: row[1] for row in rows}

                # 计算总市值
                total_market_value = 0
                for stock_id, position in cur_positions.items():
                    if stock_id in stock_prices:
                        current_price = stock_prices[stock_id]
                        shares = position["shares"]
                        market_value = current_price * shares
                        total_market_value += market_value

                # 计算总资产
                total_value = current_cash + total_market_value

                # 获取原有的stock_returns用于计算成本价
                cursor.execute(
                    "SELECT stock_returns FROM Profiles WHERE user_id = ?", (user_id,)
                )
                row = cursor.fetchone()
                old_stock_returns = json.loads(row[0]) if row else {}

                # 更新持仓比例和计算股票收益
                stock_returns = {}
                for stock_id, position in cur_positions.items():
                    if stock_id in stock_prices:
                        current_price = stock_prices[stock_id]
                        shares = position["shares"]
                        market_value = current_price * shares
                        position["ratio"] = round((market_value / total_value) * 100, 1)

                        # 获取该股票当日的交易记录
                        stock_trades = user_trades[
                            user_trades["stock_code"] == stock_id
                        ]

                        if not stock_trades.empty:  # 当天有交易
                            # 计算原有持仓数量
                            trade_shares = (
                                stock_trades["executed_quantity"].sum()
                                if stock_trades["direction"].iloc[0] == "buy"
                                else -stock_trades["executed_quantity"].sum()
                            )
                            old_shares = shares - trade_shares

                            if (
                                old_shares > 0 and stock_id in old_stock_returns
                            ):  # 有原有持仓
                                # 原有持仓部分用原profit反推成本价
                                old_profit = old_stock_returns[stock_id]["profit"] / 100
                                # 使用前一天的价格反推成本价
                                cursor.execute(
                                    "SELECT close_price FROM StockData WHERE stock_id = ? AND date = ?",
                                    (stock_id, previous_date),
                                )
                                prev_price_row = cursor.fetchone()
                                prev_price = (
                                    prev_price_row[0]
                                    if prev_price_row
                                    else current_price
                                )
                                old_cost_price = prev_price / (1 + old_profit)

                                if trade_shares > 0:  # 买入
                                    # 新买入部分的成本
                                    new_cost = (
                                        stock_trades["executed_price"]
                                        .multiply(stock_trades["executed_quantity"])
                                        .sum()
                                    )
                                    # 加权平均成本价
                                    cost_price = (
                                        old_cost_price * old_shares + new_cost
                                    ) / shares
                                else:  # 卖出
                                    cost_price = old_cost_price
                            else:  # 全新买入
                                new_cost = (
                                    stock_trades["executed_price"]
                                    .multiply(stock_trades["executed_quantity"])
                                    .sum()
                                )
                                cost_price = new_cost / shares
                        else:  # 当天没有交易
                            if stock_id in old_stock_returns:
                                # 使用前一天的价格反推成本价
                                cursor.execute(
                                    "SELECT close_price FROM StockData WHERE stock_id = ? AND date = ?",
                                    (stock_id, previous_date),
                                )
                                prev_price_row = cursor.fetchone()
                                prev_price = (
                                    prev_price_row[0]
                                    if prev_price_row
                                    else current_price
                                )
                                old_profit = old_stock_returns[stock_id]["profit"] / 100
                                cost_price = prev_price / (1 + old_profit)
                            else:
                                # 这种情况理论上不应该发生，因为没有交易的股票必然有历史记录
                                cost_price = current_price

                        # 计算收益率
                        profit_rate = ((current_price / cost_price) - 1) * 100

                        stock_returns[stock_id] = {
                            "profit": round(profit_rate, 1),
                            "market_value": round(market_value, 2),
                        }

                # 计算总收益和收益率
                total_return = total_value - ini_cash
                return_rate = (total_value / ini_cash - 1) * 100

                # 修改获取昨日收益数据的逻辑
                stock_ids = list(cur_positions.keys())
                if stock_ids:
                    placeholders = ",".join(["?" for _ in stock_ids])
                    # 直接使用当日的pct_chg
                    query = f"""
                        SELECT stock_id, pct_chg
                        FROM StockData 
                        WHERE date = ? AND stock_id IN ({placeholders})
                    """
                    params = [current_date] + stock_ids
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    yest_returns = {row[0]: round(float(row[1]), 2) for row in rows}
                else:
                    yest_returns = {}

                # 更新数据库
                cursor.execute(
                    """
                    INSERT INTO Profiles (
                        user_id,
                        gender,
                        location,
                        user_type,
                        bh_disposition_effect_category,
                        bh_lottery_preference_category,
                        bh_total_return_category,
                        bh_annual_turnover_category,
                        bh_underdiversification_category,
                        trade_count_category,
                        sys_prompt,
                        prompt,
                        self_description,
                        trad_pro,
                        fol_ind,
                        ini_cash,
                        initial_positions,
                        current_cash,
                        cur_positions,
                        total_value,
                        total_return,
                        return_rate,
                        strategy,
                        stock_returns,
                        yest_returns,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        profile_dict["gender"],
                        profile_dict["location"],
                        profile_dict["user_type"],
                        profile_dict["bh_disposition_effect_category"],
                        profile_dict["bh_lottery_preference_category"],
                        profile_dict["bh_total_return_category"],
                        profile_dict["bh_annual_turnover_category"],
                        profile_dict["bh_underdiversification_category"],
                        profile_dict["trade_count_category"],
                        profile_dict["sys_prompt"],
                        profile_dict["prompt"],
                        profile_dict["self_description"],
                        profile_dict["trad_pro"],
                        profile_dict["fol_ind"],
                        profile_dict["ini_cash"],
                        profile_dict["initial_positions"],
                        round(current_cash, 2),
                        json.dumps(cur_positions),
                        round(total_value, 2),
                        round(total_return, 2),
                        round(return_rate, 1),
                        profile_dict["strategy"],
                        json.dumps(stock_returns),
                        json.dumps(yest_returns),
                        f"{current_date} 00:00:00",  # 当前交易日
                    ),
                )

                conn.commit()

    except Exception as e:
        print(f"更新用户档案时发生错误: {str(e)}")
        raise


def generate_stock_data(decisions, df_stock, current_date):
    # 提取唯一的 stock_code 列表
    stock_codes = {decision["stock_code"] for decision in decisions}

    # 从 df_stock 中提取数据到 stock_data
    stock_data = {}
    for stock_code in stock_codes:
        # 获取最新的收盘价
        selected_row = df_stock.loc[
            (df_stock["stock_id"] == stock_code)
            & (df_stock["date"] <= pd.to_datetime(current_date) - pd.Timedelta(days=1))
        ].sort_values("date", ascending=False)
        stock_data[stock_code] = float(selected_row.iloc[0]["close_price"])

    return stock_data


import json


def read_json(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # 统计变量
    total_users = len(data)
    users_without_decisions = 0
    users_with_empty_decisions = 0
    buy_orders = 0
    sell_orders = 0

    converted_decisions = []
    for user_id, user_data in data.items():
        # 检查 user_data 是否包含 stock_decisions
        if not user_data or "stock_decisions" not in user_data:
            users_without_decisions += 1
            continue

        stock_decisions = user_data["stock_decisions"]
        # 检查 stock_decisions 是否为空
        if not stock_decisions:
            users_with_empty_decisions += 1
            continue

        for stock_code, stock_info in stock_decisions.items():
            # 检查是否有 sub_orders
            if "sub_orders" not in stock_info or not stock_info["sub_orders"]:
                continue  # 如果没有 sub_orders，跳过

            # 遍历 sub_orders
            for sub_order in stock_info["sub_orders"]:
                decision = {
                    "user_id": f"{user_id}_{stock_code}",
                    "stock_code": stock_code,
                    "direction": stock_info["action"],
                    "amount": int(sub_order["quantity"]),
                    "target_price": round(sub_order["price"], 2),
                }
                converted_decisions.append(decision)

                # 统计买卖单数量
                if stock_info["action"] == "buy":
                    buy_orders += 1
                elif stock_info["action"] == "sell":
                    sell_orders += 1

    # 计算统计信息
    users_with_valid_decisions = (
        total_users - users_without_decisions - users_with_empty_decisions
    )
    decision_rate = (
        (users_with_valid_decisions / total_users) * 100 if total_users > 0 else 0
    )

    # 打印统计信息
    print("\n=== 交易决策统计 ===")
    print(f"总用户数: {total_users}")
    print(f"无决策用户数: {users_without_decisions}")
    print(f"空决策用户数: {users_with_empty_decisions}")
    print(f"有效决策用户数: {users_with_valid_decisions}")
    print(f"决策覆盖率: {decision_rate:.2f}%")
    print("\n=== 交易订单统计 ===")
    print(f"买入订单数: {buy_orders}")
    print(f"卖出订单数: {sell_orders}")
    print(f"总订单数: {buy_orders + sell_orders}")
    print("==================\n")

    return converted_decisions


def test_matching_system(
    current_date: str,
    json_file_path: str = None,
    db_path: str = None,
    base_path: str = ".",
):
    """
    交易撮合系统主测试函数

    该函数是整个交易撮合引擎的主入口，负责协调所有子模块完成一个完整的
    交易日处理流程。从读取用户决策到更新数据库，实现端到端的交易处理。

    主要处理流程：
    1. 路径配置和目录创建
    2. 数据库连接和数据加载
    3. 用户交易决策读取和解析
    4. 订单撮合和交易执行
    5. 交易结果统计和保存
    6. 数据库更新（股票数据、交易明细、用户档案）
    7. 异常情况处理（如无交易时的节假日模式）

    Args:
        current_date (str): 当前交易日期，格式为'YYYY-MM-DD'
        json_file_path (str, optional): 用户决策JSON文件路径，默认使用标准路径
        db_path (str, optional): 数据库文件路径，默认使用标准路径
        base_path (str): 项目根目录路径，默认为当前目录

    Note:
        - 支持交易日和非交易日两种模式
        - 包含完整的异常处理和错误日志
        - 会自动创建必要的输出目录
        - 支持大规模用户并发交易处理

    Example:
        >>> test_matching_system(
        ...     current_date='2023-06-15',
        ...     base_path='./logs',
        ...     db_path='./data/user.db'
        ... )
    """
    # 设置默认路径

    if json_file_path is None:
        json_file_path = f"logs/2023-06-15.json"
    if db_path is None:
        db_path = f"{base_path}/data/sys_100.db"

    # 创建simulation_results目录
    output_dir = f"{base_path}/simulation_results/{current_date}"
    os.makedirs(output_dir, exist_ok=True)

    # 连接数据库 读取交易数据和股票信息
    conn = sqlite3.connect(db_path)
    df_stock = pd.read_sql_query(f"SELECT * FROM StockData;", conn)
    df_stock_profile = pd.read_sql_query(f"SELECT * FROM StockProfile;", conn)
    df_stock["date"] = pd.to_datetime(df_stock["date"])
    df_stock_sim = df_stock.copy(deep=True)
    df_stock_profile_real = df_stock_profile.copy(deep=True)
    conn.close()

    try:
        # 读取决策
        print(f"xxxxxxxxxxxxxxxx当前时间为{current_date}xxxxxxxxxxxxxxxxxxxxx")
        test_decisions = read_json(json_file_path)
        # 如果全部hold或者全部失败，当做节假日处理
        if len(test_decisions) > 0:
            # 生成股票数据
            stock_data = generate_stock_data(test_decisions, df_stock_sim, current_date)
            # 处理某一天
            process_trading_day(
                test_decisions,
                stock_data,
                current_date,
                output_dir,
                db_path,
                df_stock_sim,
                df_stock_profile_real,
                json_file_path,
            )
        else:
            update_profiles_table_holiday(current_date, db_path)
            update_stock_data_table_holiday(
                current_date=current_date, db_path=db_path, df_stock=df_stock_sim
            )

        print("\n===撮合交易完成===")
        # print(f'所有文件均保存到本地{output_dir}路径下')
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback

        print(f"详细错误信息:\n{traceback.format_exc()}")


def update_stock_data_table_holiday(
    current_date: str,
    db_path: str = "data/sys_100.db",
    real_data_path: str = "data/stock_data.csv",
    df_stock: pd.DataFrame = None,
):
    """
    更新数据库中的 StockData 表。

    Args:
        results (dict): 交易结果字典，包含每支股票的收盘价、成交量等信息。
        current_date (str): 当前交易日期，格式为 'YYYY-MM-DD'。
        db_path (str): 数据库文件路径，默认为 'sys_100.db'。
        output_dir (str): 输出目录，默认为 'simulation_results/{current_date}'。
    """
    try:

        current_date = pd.to_datetime(current_date)

        # 读取真实的历史数据（包含到2025年的数据）
        if not os.path.exists(real_data_path):
            raise FileNotFoundError(f"历史交易数据文件 {real_data_path} 不存在")
        historical_data_real = pd.read_csv(real_data_path)
        historical_data_real["date"] = pd.to_datetime(historical_data_real["date"])

        # 连接数据库
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 获取所有股票代码--全部上证50
            all_stock_codes = df_stock["stock_id"].unique()

            # 更新每支股票的数据
            for stock_code in all_stock_codes:
                # 获取当日的真实交易数据作为基准
                current_real_data = (
                    historical_data_real[
                        (historical_data_real["ts_code"] == stock_code)
                        & (historical_data_real["date"] <= current_date)
                    ]
                    .sort_values("date", ascending=False)
                    .iloc[0]
                )

                # 获取该股票交易数据（模拟）
                stock_history_sim = df_stock[
                    df_stock["stock_id"] == stock_code
                ].sort_values("date", ascending=False)
                prev_data = stock_history_sim[
                    stock_history_sim["date"] < current_date
                ].iloc[0]
                prev_close = prev_data["close_price"]

                large_order_net = 0  # 没有交易则大单资金流为0
                # 使用前一天的指标
                pct_change = current_real_data["pct_chg"]
                closing_price = prev_data["close_price"] * (1 + pct_change / 100)
                volume = 10000
                price_change = closing_price - prev_data["close_price"]

                # 使用当日真实数据作为基准，根据模拟收盘价等比例计算估值指标
                price_ratio = closing_price / current_real_data["close"]
                new_pe_ttm = current_real_data["pe_ttm"] * price_ratio
                new_pb = current_real_data["pb"] * price_ratio
                new_ps_ttm = current_real_data["ps_ttm"] * price_ratio
                new_dv_ttm = current_real_data["dv_ttm"] / price_ratio

                # 添加最新的收盘价和成交量
                new_row = pd.DataFrame(
                    {
                        "stock_id": [stock_code],
                        "close_price": [closing_price],
                        "pre_close": [prev_close],
                        "change": [price_change],
                        "pct_chg": [pct_change],
                        "pe_ttm": [new_pe_ttm],
                        "pb": [new_pb],
                        "ps_ttm": [new_ps_ttm],
                        "dv_ttm": [new_dv_ttm],
                        "vol": [volume],
                        "vol_5": [None],
                        "vol_10": [None],
                        "vol_30": [None],
                        "ma_hfq_5": [None],
                        "ma_hfq_10": [None],
                        "ma_hfq_30": [None],
                        "elg_amount_net": [large_order_net],
                        "date": [pd.to_datetime(current_date)],
                    }
                )
                stock_history_sim = pd.concat(
                    [stock_history_sim, new_row], ignore_index=True
                )
                stock_history_sim = stock_history_sim.sort_values(
                    "date", ascending=False
                )

                # 计算技术指标
                vol5 = (
                    stock_history_sim["vol"]
                    .iloc[::-1]
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                vol10 = (
                    stock_history_sim["vol"]
                    .iloc[::-1]
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                vol30 = (
                    stock_history_sim["vol"]
                    .iloc[::-1]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .iloc[-1]
                )

                ma5 = (
                    stock_history_sim["close_price"]
                    .iloc[::-1]
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ma10 = (
                    stock_history_sim["close_price"]
                    .iloc[::-1]
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ma30 = (
                    stock_history_sim["close_price"]
                    .iloc[::-1]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .iloc[-1]
                )

                if isinstance(current_date, pd.Timestamp):
                    current_date = current_date.strftime("%Y-%m-%d")
                try:
                    cursor.execute(
                        """
                    INSERT INTO StockData (
                        stock_id,
                        date,
                        close_price,
                        pre_close,
                        change,
                        pct_chg,
                        pe_ttm,
                        pb,
                        ps_ttm,
                        dv_ttm,
                        vol,
                        vol_5,
                        vol_10,
                        vol_30,
                        elg_amount_net,
                        ma_hfq_5,
                        ma_hfq_10,
                        ma_hfq_30
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            stock_code,
                            current_date,
                            round(float(closing_price), 2),
                            round(float(prev_close), 2),
                            round(float(price_change), 2),
                            round(float(pct_change), 4),
                            round(float(new_pe_ttm), 4),
                            round(float(new_pb), 4),
                            round(float(new_ps_ttm), 4),
                            round(float(new_dv_ttm), 4),
                            int(volume) if pd.notna(volume) else None,
                            round(float(vol5), 2) if pd.notna(vol5) else None,
                            round(float(vol10), 2) if pd.notna(vol10) else None,
                            round(float(vol30), 2) if pd.notna(vol30) else None,
                            round(float(large_order_net), 2),
                            round(float(ma5), 5),
                            round(float(ma10), 5),
                            round(float(ma30), 5),
                        ),
                    )
                except Exception as e:
                    print(f"插入数据时发生错误: {e}")

                # 提交更改
                conn.commit()

    except Exception as e:
        # print(f"更新股票数据时发生错误: {e}")
        raise


def update_profiles_table_holiday(current_date: str, db_path: str):
    """
    根据交易结果更新用户档案表

    Args:
        current_date: 当前交易日期
        db_path: 数据库路径
    """
    try:

        current_date_obj = pd.to_datetime(current_date)
        previous_date = (current_date_obj - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d 00:00:00"
        )

        # 连接数据库并获取所有用户信息
        with sqlite3.connect(db_path) as conn:

            cursor = conn.cursor()
            # 获取所有用户信息
            cursor.execute(
                "SELECT * FROM Profiles where created_at = ?", (previous_date,)
            )
            profiles = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            # 处理每个用户的更新
            for profile in profiles:
                profile_dict = dict(zip(columns, profile))
                user_id = str(profile_dict["user_id"])

                # 更新数据库
                cursor.execute(
                    """
                    INSERT INTO Profiles (
                        user_id,
                        gender,
                        location,
                        user_type,
                        bh_disposition_effect_category,
                        bh_lottery_preference_category,
                        bh_total_return_category,
                        bh_annual_turnover_category,
                        bh_underdiversification_category,
                        trade_count_category,
                        sys_prompt,
                        prompt,
                        self_description,
                        trad_pro,
                        fol_ind,
                        ini_cash,
                        initial_positions,
                        current_cash,
                        cur_positions,
                        total_value,
                        total_return,
                        return_rate,
                        strategy,
                        stock_returns,
                        yest_returns,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        profile_dict["gender"],
                        profile_dict["location"],
                        profile_dict["user_type"],
                        profile_dict["bh_disposition_effect_category"],
                        profile_dict["bh_lottery_preference_category"],
                        profile_dict["bh_total_return_category"],
                        profile_dict["bh_annual_turnover_category"],
                        profile_dict["bh_underdiversification_category"],
                        profile_dict["trade_count_category"],
                        profile_dict["sys_prompt"],
                        profile_dict["prompt"],
                        profile_dict["self_description"],
                        profile_dict["trad_pro"],
                        profile_dict["fol_ind"],
                        profile_dict["ini_cash"],
                        profile_dict["initial_positions"],
                        profile_dict["current_cash"],
                        profile_dict["cur_positions"],
                        profile_dict["total_value"],
                        profile_dict["total_return"],
                        profile_dict["return_rate"],
                        profile_dict["strategy"],
                        profile_dict["stock_returns"],
                        profile_dict["yest_returns"],
                        f"{current_date} 00:00:00",  # 当前交易日
                    ),
                )

                conn.commit()

    except Exception as e:
        # print(f"更新用户档案时发生错误: {str(e)}")
        raise


# test_matching_system(
#                 current_date='2023-06-15',
#                 base_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/logs_100_0127_gemini_exp_1206',
#                 db_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/logs_100_0127_gemini_exp_1206/user_100.db',
#                 json_file_path='/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/存档言哥提到之后/0127/logs_100_0127_ds/trading_records/2023-06-15.json'
#             )

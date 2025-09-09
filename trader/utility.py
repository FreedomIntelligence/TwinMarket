from typing import Optional
import sqlite3
from typing import Union, Dict, List
import logging
from typing import Dict, List, Union
from typing import Union, List, Dict
import os
import yaml
import json
import re
import requests
import random
from Agent import BaseAgent
import pandas as pd

# 全局 logger 实例
_logger = None

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Investment Decision Schema",
    "description": "Schema for investment decisions based on analysis and beliefs.",
    "type": "object",
    "required": [
        "analysis",
        "decision",
        "amount",
        "target_price",
        "belief"
    ],
    "properties": {
        "分析过程": {
            "type": "object",
            "required": [
                "引用的新闻或公告",
                "价格信息分析",
                "市场趋势和情绪",
                "投资风格和人设"
            ],
            "properties": {
                "引用的新闻或公告": {
                    "type": "string",
                    "description": "Summary of relevant news or announcements used for analysis."
                },
                "价格信息分析": {
                    "type": "string",
                    "description": "Analysis of price information, including current price, trend, etc."
                },
                "市场趋势和情绪": {
                    "type": "string",
                    "description": "Assessment of market trends and overall sentiment."
                },
                "投资风格和人设": {
                    "type": "string",
                    "description": "Description of the investor's style and risk profile."
                }
            }
        },
        "决策": {
            "type": "object",
            "description": "Investment decision for each stock.",
            "additionalProperties": {
                "type": "string",
                "enum": ["buy", "sell", "hold"]
            }
        },
        "数量": {
            "type": "object",
            "description": "Amount of shares for each stock.",
            "additionalProperties": {
                "type": "integer",
                "minimum": 0
            }
        },
        "目标价格": {
            "type": "object",
            "description": "Target price for each stock.",
            "additionalProperties": {
                "type": "number",
                "minimum": 0
            }
        },
        "信念": {
            "type": "object",
            "required": [
                "市场趋势",
                "市场估值",
                "经济状况",
                "市场情绪",
                "自我评价"
            ],
            "properties": {
                "市场趋势": {
                    "type": "string",
                    "description": "Belief about the future market trend."
                },
                "市场估值": {
                    "type": "string",
                    "description": "Belief about the current market valuation."
                },
                "经济状况": {
                    "type": "string",
                    "description": "Belief about the overall economic outlook."
                },
                "市场情绪": {
                    "type": "string",
                    "description": "Belief about the current market sentiment."
                },
                "自我评价": {
                    "type": "string",
                    "description": "Self-assessment of the investor's abilities and performance."
                }
            }
        }
    }}

# STOCK_DATA_PATH='/home/export/base/ycsc_wangbenyou/zhangyf/online1/AI_stock_market/test_agent_zyf/stock_data.csv'
STOCK_PROFILE_PATH = 'data/UserDB_zyf_0109/stock_profile.csv'
STOCK_PROFILE_PATH2 = 'data/stock_profile.csv'
STOCK_PROFILE_DICT={
'TLEI': '该指数为交通与运输指数，包含2支成分股，包括中远海控(SH601919, 权重52.34%)、中国船舶(SH600150, 权重47.66%)。',
'MEI': '该指数为制造业指数，包含8支成分股，包括隆基绿能(SH601012, 权重16.51%)、海尔智家(SH600690, 权重15.78%)、三一重工(SH600031, 权重15.29%)、国电南瑞(SH600406, 权重14.6%)、上汽集团(SH600104, 权重11.98%)、通威股份(SH600438, 权重10.92%)、特变电工(SH600089, 权重10.12%)、长城汽车(SH601633, 权重4.8%)。',
'CPEI': '该指数为化工与制药指数，包含3支成分股，包括恒瑞医药(SH600276, 权重45.93%)、万华化学(SH600309, 权重28.39%)、药明康德(SH603259, 权重25.68%)。',
'IEEI': '该指数为基础设施与工程指数，包含3支成分股，包括中国建筑(SH601668, 权重52.25%)、中国中铁(SH601390, 权重27.64%)、中国电建(SH601669, 权重20.11%)。',
'REEI': '该指数为房地产指数，包含1支成分股，包括保利发展(SH600048, 权重100.0%)。',
'TSEI': '该指数为旅游与服务指数，包含1支成分股，包括中国中免(SH601888, 权重100.0%)。',
'CGEI': '该指数为消费品指数，包含5支成分股，包括贵州茅台(SH600519, 权重69.18%)、伊利股份(SH600887, 权重13.13%)、山西汾酒(SH600809, 权重7.17%)、海天味业(SH603288, 权重5.44%)、片仔癀(SH600436, 权重5.08%)。',
'TTEI': '该指数为科技与通信指数，包含10支成分股，包括中芯国际(SH688981, 权重18.07%)、海光信息(SH688041, 权重11.89%)、中国电信(SH601728, 权重10.21%)、中国联通(SH600050, 权重9.98%)、中微公司(SH688012, 权重9.79%)、中国移动(SH600941, 权重9.76%)、中国核电(SH601985, 权重9.05%)、韦尔股份(SH603501, 权重8.53%)、金山办公(SH688111, 权重6.92%)、兆易创新(SH603986, 权重5.81%)。',
'EREI': '该指数为能源与资源指数，包含6支成分股，包括长江电力(SH600900, 权重33.45%)、紫金矿业(SH601899, 权重25.88%)、中国神华(SH601088, 权重13.19%)、中国石化(SH600028, 权重9.29%)、中国石油(SH601857, 权重9.12%)、陕西煤业(SH601225, 权重9.07%)。',
'FSEI': '该指数为金融服务指数，包含11支成分股，包括中国平安(SH601318, 权重22.86%)、招商银行(SH600036, 权重17.94%)、中信证券(SH600030, 权重11.97%)、兴业银行(SH601166, 权重10.47%)、工商银行(SH601398, 权重8.6%)、交通银行(SH601328, 权重8.04%)、农业银行(SH601288, 权重6.12%)、中国太保(SH601601, 权重4.63%)、中国银行(SH601988, 权重4.21%)、中国人寿(SH601628, 权重2.8%)、邮储银行(SH601658, 权重2.35%)。',
'CSI300': '该指数为沪深300指数，包含300支成分股，由沪深市场中规模大、流动性好的最具代表性的300只证券组成。'
 }

INDICATORS = [
    "name",
    "reg_capital",
    "setup_date",
    "introduction",
    "business_scope",
    "employees",
    "main_business",
    "city",
    "industry",
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
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
]

MAPPING_DICT = {
    'pe_ttm': '市盈率(TTM)',
    'pb': '市净率',
    'ps_ttm': '市销率(TTM)',
    'dv_ttm': '股息率(TTM)',
    'vol_5': '5日平均交易额',
    'vol_10': '10日平均交易额',
    'vol_30': '30日平均交易额',
    'ma_hfq_10': '10日移动平均线(后复权)',
    'ma_hfq_30': '30日移动平均线(后复权)',
    'ma_hfq_5': '5日移动平均线(后复权)',
    'macd_hfq': 'MACD柱状线(后复权)',
    'macd_dea_hfq': 'MACD慢线(后复权)',
    'macd_dif_hfq': 'MACD快线(后复权)',
    'elg_amount_net': '超大单资金净流入',
    'ts_code': '股票代码',
    'stock_id': '股票代码',
    'reg_capital': '注册资本',
    'setup_date': '成立日期',
    'introduction': '公司简介',
    'business_scope': '经营范围',
    'employees': '员工人数',
    'main_business': '主营业务',
    'city': '所在城市',
    'name': '公司名称',
    'industry': '所属行业'
}

MAPPING_INDICATORS={
    '基本面': ['pe_ttm', 'pb', 'ps_ttm', 'dv_ttm'],
    '技术面': ['vol_5', 'vol_10', 'vol_30', 'ma_hfq_5', 'ma_hfq_10', 'ma_hfq_30', 'macd_dif_hfq', 'macd_dea_hfq', 'macd_hfq', 'elg_amount_net'],
    '宏观指标': ['reg_capital', 'setup_date', 'introduction', 'business_scope', 'employees', 'main_business', 'city', 'industry'],
    '混合': ['pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'vol_5', 'vol_10', 'vol_30', 'ma_hfq_5', 'ma_hfq_10', 'ma_hfq_30', 'macd_dif_hfq', 'macd_dea_hfq', 'macd_hfq', 'elg_amount_net']
}

MAPPING_INDICATORS2={
    '基本面': ['pe_ttm', 'pb'],
    '技术面': ['vol_5', 'vol_10', 'vol_30', 'ma_hfq_5', 'ma_hfq_10', 'elg_amount_net'],
}

MAPPING_INDICATORS3={
    '基本面': ['pe_ttm', 'pb', 'ps_ttm', 'dv_ttm'],
    '技术面': ['vol_5', 'vol_10', 'vol_30', 'ma_hfq_5', 'ma_hfq_10', 'elg_amount_net'],
}

# 原始映射
GO = {
    'YF01': '交通与运输指数',
    'YF02': '制造业指数',
    'YF03': '化工与制药指数',
    'YF04': '基础设施与工程指数',
    'YF05': '房地产指数',
    'YF06': '旅游与服务指数',
    'YF07': '消费品指数',
    'YF08': '科技与通信指数',
    'YF09': '能源与资源指数',
    'YF10': '金融服务指数'
}

# 反向映射
BACK ={'交通与运输指数': 'YF01',
 '制造业指数': 'YF02',
 '化工与制药指数': 'YF03',
 '基础设施与工程指数': 'YF04',
 '房地产指数': 'YF05',
 '旅游与服务指数': 'YF06',
 '消费品指数': 'YF07',
 '科技与通信指数': 'YF08',
 '能源与资源指数': 'YF09',
 '金融服务指数': 'YF10'}

def convert_str_to_number(value):
    """
    将字符串转换为数字（float 或 int），如果转换失败则返回 None。

    Args:
        value (str): 需要转换的字符串

    Returns:
        float or int or None: 转换后的数字，如果转换失败则返回 None
    """
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            # 尝试转换为浮点数
            return float(value)
        except ValueError:
            # 如果转换失败，返回 None
            return None
    return None


def preprocess_stock_decisions(stock_decisions: dict) -> dict:
    """
    预处理 stock_decisions，将字符串字段转换为数字。

    Args:
        stock_decisions (dict): 包含股票决策的字典

    Returns:
        dict: 处理后的股票决策字典
    """
    for stock_code, decision_list in stock_decisions.items():
        # 将 decision_list 转换为字典
        decision_dict = {}
        for item in decision_list:
            if isinstance(item, dict):
                decision_dict.update(item)

        # 转换 cur_position、target_position 和 target_price
        for field in ["cur_position", "target_position", "target_price"]:
            if field in decision_dict:
                decision_dict[field] = convert_str_to_number(decision_dict[field])

        # 更新决策信息
        stock_decisions[stock_code] = decision_dict

    return stock_decisions


def parse_response_yaml(response: str, max_retries: int = 3, log_dir: str = "./", debug: bool = False, prompt: str = None) -> Union[Dict, List[Dict]]:
    """
    解析 LLM 返回的响应，支持 YAML 对象或 YAML 数组。

    Args:
        response (str): LLM 返回的响应内容。
        max_retries (int): 最大重试次数，默认为 3。
        log_dir (str): 日志目录路径，默认为 "logs"。
        debug (bool): 是否启用调试模式，默认为 False。
        prompt (str): 用户提供的提示内容，可选。

    Returns:
        Union[Dict, List[Dict]]: 解析后的 YAML 对象或 YAML 数组。

    Raises:
        ValueError: 如果解析失败且达到最大重试次数。
    """
    # # # 确保日志目录存在
    # os.makedirs(log_dir, exist_ok=True)

    # # 错误日志文件路径（只记录错误信息和错误的 YAML 内容）
    # parse_error_log = os.path.join(log_dir, "parse_error.log")

    # # 创建专门用于错误日志记录的 logger
    # error_logger = logging.getLogger("parse_error_logger")
    # error_logger.setLevel(logging.ERROR)
    # file_handler = logging.FileHandler(parse_error_log, mode='a', encoding='utf-8')
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    # error_logger.addHandler(file_handler)

    fixAgent = BaseAgent(config_path="./config/api.yaml")
    retries = 0

    while retries <= max_retries:
        # 尝试提取 ```yaml 块中的内容
        yaml_match = re.search(r'```yaml\s*([\s\S]*?)\s*```', response, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
        else:
            # 如果没有找到 ```yaml 块，假设整个响应是 YAML
            yaml_content = response

        try:
            # 解析 YAML 内容
            yaml_content = preprocess_yaml(yaml_content)
            parsed_yaml = yaml.safe_load(yaml_content)

            # 统一转换为小写键（如果是字典）
            if isinstance(parsed_yaml, dict):
                parsed_yaml = {k: v for k, v in parsed_yaml.items()}
            elif isinstance(parsed_yaml, list):
                parsed_yaml = [
                    {k: v for k, v in item.items()} if isinstance(item, dict) else item
                    for item in parsed_yaml
                ]

            return parsed_yaml
        except yaml.YAMLError as e:
            # 如果启用了调试模式，打印错误信息
            if debug:
                print(f"\033[91mYAML Parse Error: {str(e)}\033[0m")
                print(f"Original Input: {yaml_content}")

            if retries == max_retries:
                # 记录错误日志：只记录错误信息和出错的原始 YAML 内容
                print_debug(f"Failed to parse YAML after {max_retries} retries.\n"
                                   f"Error: {str(e)}\n"
                                   f"Original YAML:\n{yaml_content}\n", debug=True)
                raise ValueError(f"Failed to parse YAML after {max_retries} retries.")

            # 调用 fixAgent 修复 YAML 内容（不打印调试信息）
            response = fixAgent.get_response(
                user_input=(
                    f"Fix the following YAML content which failed to parse with error: {str(e)}\n\n"
                    f"{yaml_content}\n\n"
                    "Please ensure that all existing keys are preserved in the corrected YAML."
                    f"\n{prompt if prompt is not None else ''}\n"
                    "The corrected YAML should be wrapped in a ```yaml code block like this:\n"
                    "```yaml\n"
                    "key: value\n"
                    "```"
                ),
                system_prompt=None,
                temperature=0.0,
            )
            response = response.get("response")
            retries += 1

    raise ValueError(f"Failed to parse YAML after {max_retries} retries.")


def preprocess_yaml(yaml_content: str) -> str:
    """
    Preprocess the YAML content to clean up the string following the keyword 'reason'.
    Removes problematic characters like newlines, extra spaces after 'reason:', and all quotes.

    Args:
        yaml_content (str): The raw YAML content as a string.

    Returns:
        str: The processed YAML content as a string.
    """
    def clean_reason(match):
        # Extract the matched reason string and clean it
        reason_content = match.group(1)
        # Remove newlines, tabs, and strip extra spaces
        cleaned_reason = " ".join(reason_content.split())
        return f"reason: {cleaned_reason}"

    # Regex to find 'reason:' followed by any content and clean it
    yaml_content = re.sub(r'reason:\s*(.*)', clean_reason, yaml_content)

    # Replace all kinds of quotes (both Chinese and English) with an empty string
    yaml_content = re.sub(r'[“”""]', '', yaml_content)

    return yaml_content


def parse_response_json(response: str, max_retries: int = 3, log_file: str = "logs/parse_error.log") -> Union[Dict, List[Dict]]:
    """
    解析 LLM 返回的响应，支持 JSON 对象或 JSON 数组。

    Args:
        response (str): LLM 返回的响应内容。
        max_retries (int): 最大重试次数，默认为 3。
        log_file (str): 日志文件路径，默认为 "log/parse_error.log"。

    Returns:
        Union[Dict, List[Dict]]: 解析后的 JSON 对象或 JSON 数组。

    Raises:
        ValueError: 如果解析失败且达到最大重试次数。
    """
    fixAgent = BaseAgent()
    retries = 0

    while retries <= max_retries:
        # 尝试提取 ```json 块中的内容
        json_match = re.search(r'```json\s*(\[.*?\]|{.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        else:
            # 如果没有找到 ```json 块，假设整个响应是 JSON
            json_content = response

        # 预处理 JSON 内容
        json_content = preprocess_json(json_content)

        try:
            # 解析 JSON 内容
            parsed_json = json.loads(json_content)

            # 统一转换为小写键（如果是字典）
            if isinstance(parsed_json, dict):
                parsed_json = {k.lower(): v for k, v in parsed_json.items()}
            elif isinstance(parsed_json, list):
                # 如果是列表，确保每个元素是字典并统一转换为小写键
                parsed_json = [
                    {k.lower(): v for k, v in item.items()} if isinstance(item, dict) else item
                    for item in parsed_json
                ]

            return parsed_json
        except json.JSONDecodeError as e:
            # 只有在解析失败时才打印错误信息和原始输入
            print(f"\033[91mJSON Decode Error: {str(e)}\033[0m")
            print(f"Original Input: {response}")

            if retries == max_retries:
                # 记录错误日志
                logging.basicConfig(
                    filename=log_file,
                    level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                logging.error(f"Failed to parse JSON after {max_retries} retries.\n"
                              f"Error: {str(e)}\n"
                              f"Original Input: {response}")
                raise ValueError(f"Failed to parse JSON after {max_retries} retries.")

            # 调用 fixAgent 修复 JSON 内容（不打印调试信息）
            response = fixAgent.get_response(
                user_input=(
                    f"Fix the following JSON content which failed to parse with error: {str(e)}\n\n"
                    f"{json_content}\n\n"
                    "Please ensure that all existing keys are preserved in the corrected JSON."
                ),
                system_prompt=None,
                temperature=0.0,
            ).get("response")
            retries += 1

    raise ValueError(f"Failed to parse JSON after {max_retries} retries.")


def preprocess_json(json_content: str) -> str:
    json_content = re.sub(r'[“”]', '"', json_content)
    json_content = re.sub(r'，', ',', json_content)
    json_content = re.sub(r'\s+', ' ', json_content).strip()
    # json_content = re.sub(r'[‘’]', "'", json_content)
    # json_content = re.sub(r'。', '.', json_content)
    # json_content = re.sub(r'：', ':', json_content)
    # json_content = re.sub(r'；', ';', json_content)
    # json_content = re.sub(r'？', '?', json_content)
    # json_content = re.sub(r'！', '!', json_content)
    # json_content = re.sub(r'（', '(', json_content)
    # json_content = re.sub(r'）', ')', json_content)
    return json_content


def print_debug(message: str, debug: bool, log_dir: str = "logs"):
    if debug:
        print(f"\033[94m{message}\033[0m")


def setup_logger(log_file: str = "logs/simulation_debug.log", debug: bool = False) -> logging.Logger:
    """
    配置独立的日志记录器。

    Args:
        log_file (str): 日志文件的路径。默认为 "logs/simulation_debug.log"。
        debug (bool): 是否在终端显示日志。默认为 False。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    global _logger

    # 如果 logger 已经配置过，直接返回
    if _logger is not None:
        return _logger

    # 配置日志格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # 创建一个独立的日志记录器
    _logger = logging.getLogger("print_debug_logger")
    _logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 清除已有的 handlers，避免重复添加
    _logger.handlers.clear()

    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 添加文件 handler，确保日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    _logger.addHandler(file_handler)

    # 如果 debug 为 True，添加终端 handler
    if debug:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(log_format))
        _logger.addHandler(stream_handler)

    return _logger


# def print_debug(message: str, debug: bool, log_dir: str = "logs"):
#     """
#     使用独立的日志记录器记录日志，并在 debug 为 True 时在终端显示。

#     Args:
#         message (str): 要记录和显示的日志消息。
#         debug (bool): 是否在终端显示日志。
#         log_dir (str): 日志文件的目录。默认为 "logs"。
#     """
#     # 配置独立的日志记录器
#     log_file = os.path.join(log_dir, "simulation_debug.log")
#     logger = setup_logger(log_file=log_file, debug=debug)

#     # 记录日志
#     logger.info(message)


def merge_nested_lists(dict1, dict2):

    # 创建结果字典，首先复制dict1的非data字段
    result = dict1.copy()

    # 特殊处理data字段的合并
    if 'data' in dict1 and 'data' in dict2:
        # 使用日期作为匹配键
        merged_data = []

        # 创建一个映射，以日期为键
        dict1_map = {item['date']: item for item in dict1['data']}
        dict2_map = {item['date']: item for item in dict2['data']}

        # 合并所有唯一的日期
        all_dates = sorted(set(dict1_map.keys()) | set(dict2_map.keys()))

        for date in all_dates:
            # 合并对应日期的数据
            merged_dict = dict1_map.get(date, {}).copy()
            merged_dict.update(dict2_map.get(date, {}))
            merged_data.append(merged_dict)

        result['data'] = merged_data

    # 更新其他非data字段
    result.update({k: v for k, v in dict2.items() if k != 'data'})

    return result

# TODO


def convert_values_to_float(decision_args):
    try:
        if 'stock_decisions' in decision_args:
            # 处理可能的嵌套字典情况
            stock_decisions = decision_args['stock_decisions']
            if isinstance(stock_decisions, dict):
                # 如果stock_decisions本身是字典
                for stock_id, decision in stock_decisions.items():
                    if isinstance(decision, dict):
                        # 转换数值为float
                        for key in ['trading_position', 'target_price']:
                            if key in decision:
                                try:
                                    decision[key] = float(decision[key])
                                except (ValueError, TypeError):
                                    print(f"无法转换 {stock_id} 的 {key} 值为float")
            elif isinstance(stock_decisions, set):
                # 如果stock_decisions是集合,转换为字典
                stock_dict = stock_decisions.pop() if stock_decisions else {}
                if isinstance(stock_dict, dict):
                    for stock_id, decision in stock_dict.items():
                        if isinstance(decision, dict):
                            for key in ['trading_position', 'target_price']:
                                if key in decision:
                                    try:
                                        decision[key] = float(decision[key])
                                    except (ValueError, TypeError):
                                        print(f"无法转换 {stock_id} 的 {key} 值为float")
                    decision_args['stock_decisions'] = stock_dict

        return decision_args
    except Exception as e:
        raise ValueError(f"生成的持仓和目标价格必须是数字")


def rerank_documents(query, documents, timelines, top_n=2):
    """
    使用重排模型对文档进行重排序

    参数:
        query (str): 查询文本
        documents (list): 待重排的文档列表
        timelines (list): 文档对应的时间列表
        top_n (int): 返回前n个结果


    返回:
        dict: API响应结果
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '../config/reranker.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    api_key = random.choice(config['api_key'])
    model_name = config['model_name']
    base_url = config['base_url']

    payload = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
        "max_chunks_per_doc": 1024,
        "overlap_tokens": 80
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", base_url, json=payload, headers=headers)
    results = response.json()['results']

    # 返回top_n个结果
    selected_docs = []
    selected_times = []
    for result in results[:top_n]:
        idx = result['index']
        selected_docs.append(documents[idx])
        selected_times.append(timelines[idx])

    return selected_docs, selected_times


async def rerank_documents_async(query, documents, timelines, top_n=2):
    """
    使用重排模型对文档进行异步重排序

    参数:
        query (str): 查询文本
        documents (list): 待重排的文档列表
        timelines (list): 文档对应的时间列表
        top_n (int): 返回前n个结果

    返回:
        tuple: (selected_docs, selected_times) 重排序后的文档和对应时间
    """
    import aiohttp

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '../config/reranker.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    api_key = random.choice(config['api_key'])
    model_name = config['model_name']
    base_url = config['base_url']

    payload = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
        "max_chunks_per_doc": 1024,
        "overlap_tokens": 80
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(base_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"API request failed with status {response.status}: {error_text}")

                results = await response.json()
                results = results['results']

                # 返回top_n个结果
                selected_docs = []
                selected_times = []
                for result in results[:top_n]:
                    idx = result['index']
                    selected_docs.append(documents[idx])
                    selected_times.append(timelines[idx])

                return selected_docs, selected_times

        except aiohttp.ClientError as e:
            print(f"Error during reranking request: {str(e)}")
            # 发生错误时返回原始文档的前top_n个
            return documents[:top_n], timelines[:top_n]


def init_system(current_date: pd.Timestamp, db_path: str, forum_db: str) -> None:
    """
    初始化系统，清理数据库中超过指定日期的数据

    Args:
        current_date (pd.Timestamp): 当前日期
        db_path (str): 数据库路径
        forum_db (str): 论坛数据库路径
    """

    # 转换日期格式
    date_str = current_date.strftime('%Y-%m-%d')
    date_time_str = current_date.strftime('%Y-%m-%d 00:00:00')

    try:
        # 清理交易系统数据库
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")

            try:
                # 清理 Profiles 表
                cursor.execute("""
                    DELETE FROM Profiles 
                    WHERE created_at >= ?
                """, (date_time_str,))
                profiles_deleted = cursor.rowcount

                cursor.execute("""
                    DELETE FROM StockData 
                    WHERE date >= ?
                """, (date_str,))
                stock_data_deleted = cursor.rowcount

                cursor.execute("""
                    DELETE FROM TradingDetails 
                    WHERE date_time >= ?
                """, (date_str,))
                trading_details_deleted = cursor.rowcount

                conn.commit()
                print(f"\n=== 交易系统数据检查 ({date_str}) ===")
                print(f"Profiles表: {'无需清理' if profiles_deleted == 0 else f'删除 {profiles_deleted} 条记录'}")
                print(f"StockData表: {'无需清理' if stock_data_deleted == 0 else f'删除 {stock_data_deleted} 条记录'}")
                print(f"TradingDetails表: {'无需清理' if trading_details_deleted == 0 else f'删除 {trading_details_deleted} 条记录'}")
                print("===================\n")

            except Exception as e:
                conn.rollback()
                raise e

        # 清理论坛数据库
        with sqlite3.connect(forum_db) as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")

            try:
                # 检查表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='post_references'")
                if not cursor.fetchone():
                    raise ValueError("表 post_references 不存在，请检查数据库初始化")

                # 清理 post_references 表
                cursor.execute("""
                    DELETE FROM post_references 
                    WHERE created_at >= ?
                """, (date_time_str,))
                references_deleted = cursor.rowcount

                # 清理 posts 表
                cursor.execute("""
                    DELETE FROM posts 
                    WHERE created_at >= ?
                """, (date_time_str,))
                posts_deleted = cursor.rowcount

                # 清理 reactions 表
                cursor.execute("""
                    DELETE FROM reactions 
                    WHERE created_at >= ?
                """, (date_time_str,))
                reactions_deleted = cursor.rowcount

                conn.commit()
                print(f"=== 论坛数据检查 ({date_str}) ===")
                print(f"post_references表: {'无需清理' if references_deleted == 0 else f'删除 {references_deleted} 条记录'}")
                print(f"posts表: {'无需清理' if posts_deleted == 0 else f'删除 {posts_deleted} 条记录'}")
                print(f"reactions表: {'无需清理' if reactions_deleted == 0 else f'删除 {reactions_deleted} 条记录'}")
                print("===================\n")

            except Exception as e:
                conn.rollback()
                raise e

    except Exception as e:
        raise ValueError(f"初始化系统时发生错误: {str(e)}")

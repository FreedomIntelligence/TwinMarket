"""
行业分类字典和股票查询工具模块

该模块提供了完整的行业分类体系和股票查询功能，主要包括：
- 中英文行业分类映射字典
- 基于行业的股票查询功能
- 股票代码到行业类别的反向查询
- 多层级行业分类体系支持

功能特点：
- 支持10大行业分类体系
- 中英文双语行业名称支持
- 灵活的股票-行业映射查询
- 异常处理和数据验证

适用场景：
- 股票行业分类查询
- 投资组合行业分析
- 行业轮动策略支持
- 风险分散分析
"""

# 第三方库导入
import pandas as pd

# ============================ 行业分类字典 ============================

# 中文行业分类映射字典
# 将具体的细分行业映射到10大主要行业类别中
ch = {
    "制造业": [
        "家用电器",    # 家电制造行业
        "半导体",      # 半导体芯片行业
        "电气设备",    # 电气设备制造
        "工程机械",    # 工程机械设备
        "汽车整车"     # 汽车制造业
    ],
    "能源与资源": [
        "煤炭开采",    # 煤炭采掘业
        "水力发电",    # 水电能源
        "石油加工",    # 石油炼化
        "石油开采",    # 石油勘探开采
        "铜"          # 有色金属铜
    ],
    "金融服务": [
        "银行",        # 银行业
        "证券",        # 证券业
        "保险"         # 保险业
    ],
    "消费品": [
        "白酒",        # 白酒制造
        "乳制品",      # 乳制品加工
        "食品",        # 食品加工
        "中成药"       # 中成药制造
    ],
    "科技与通信": [
        "半导体",      # 半导体技术
        "软件服务",    # 软件开发服务
        "电信运营",    # 电信运营商
        "新型电力"     # 新能源电力
    ],
    "交通与运输": [
        "水运",        # 水上运输
        "船舶"         # 船舶制造
    ],
    "房地产": [
        "全国地产"     # 房地产开发
    ],
    "旅游与服务": [
        "旅游服务"     # 旅游服务业
    ],
    "化工与制药": [
        "化工原料",    # 化工原料
        "化学制药"     # 化学制药
    ],
    "基础设施与工程": [
        "建筑工程"     # 建筑工程
    ]
}

# 英文行业分类映射字典
# 与中文字典对应的英文版本，用于国际化支持和英文接口
eng = {
    "Manufacturing": [                    # 制造业
        "家用电器", "半导体", "电气设备", "工程机械", "汽车整车", "建筑工程"
    ],
    "Energy and Resources": [             # 能源与资源
        "煤炭开采", "水力发电", "石油加工", "石油开采", "铜"
    ],
    "Financial Services": [               # 金融服务
        "银行", "证券", "保险"
    ],
    "Consumer Goods": [                   # 消费品
        "白酒", "乳制品", "食品", "中成药"
    ],
    "Technology and Communication": [     # 科技与通信
        "半导体", "软件服务", "电信运营", "新型电力"
    ],
    "Transportation and Logistics": [     # 交通与运输
        "水运", "船舶"
    ],
    "Real Estate": [                      # 房地产
        "全国地产"
    ],
    "Tourism and Services": [             # 旅游与服务
        "旅游服务"
    ],
    "Chemical and Pharmaceuticals": [     # 化工与制药
        "化工原料", "化学制药"
    ],
    "Infrastructure and Engineering": [   # 基础设施与工程
        "建筑工程"
    ]
}


def get_stocks_by_industry(industry: str) -> list:
    """
    根据行业名称获取该行业的所有股票信息
    
    从CSV文件中查询指定行业的所有股票，返回股票代码和名称的配对信息。
    这个函数主要用于行业分析和投资组合构建。
    
    Args:
        industry (str): 要查询的行业名称（如"银行"、"半导体"等）
        
    Returns:
        list[tuple]: 股票信息元组列表，每个元组包含 (股票代码, 股票名称)
        
    Raises:
        ValueError: 当指定行业在数据中不存在时抛出异常
        FileNotFoundError: 当CSV文件不存在时抛出异常
        Exception: 其他数据处理错误
        
    Example:
        >>> stocks = get_stocks_by_industry("银行")
        >>> print(stocks)  # [('SH601398', '工商银行'), ('SH600036', '招商银行'), ...]
        
    Note:
        - 行业名称需要与CSV文件中的industry列完全匹配
        - 返回结果按CSV文件中的顺序排列
    """
    try:
        # 读取股票资料CSV文件
        df = pd.read_csv('data/xueqiu_data/stock_profile.csv')

        # 按行业筛选并获取股票代码和名称列
        filtered_df = df[df['industry'] == industry][['stock_id', 'name']]

        # 检查是否找到该行业的股票
        if filtered_df.empty:
            raise ValueError(f"行业 '{industry}' 在数据中未找到")

        # 转换为元组列表格式
        result = list(zip(filtered_df['stock_id'], filtered_df['name']))

        return result

    except FileNotFoundError:
        raise FileNotFoundError("CSV文件 'stock_profile.csv' 未找到")
    except Exception as e:
        raise Exception(f"数据处理错误: {str(e)}")


def get_stock_industry_and_category(
    stock_code: str,
    profile_path: str = "/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/data/xueqiu_data/stock_profile.csv",
) -> dict:
    """
    根据股票代码获取详细的行业和类别信息
    
    该函数实现了股票代码到行业信息的反向查询，支持两级分类体系：
    - 细分行业（industry）：具体的行业类别，如"银行"、"半导体"
    - 主要类别（category）：10大行业分类，如"金融服务"、"制造业"
    
    查询流程：
    1. 根据股票代码查找细分行业
    2. 根据细分行业匹配主要类别
    3. 返回完整的分类信息
    
    Args:
        stock_code (str): 股票代码，如"SH601398"
        profile_path (str): 股票行业数据文件路径，默认使用系统配置路径
        
    Returns:
        dict: 包含行业分类信息的字典，格式为：
            {
                "industry": str,    # 细分行业名称
                "category": str     # 主要类别名称
            }
            如果股票代码未找到，返回 {"industry": "未知", "category": "其他"}
            
    Example:
        >>> info = get_stock_industry_and_category("SH601398")
        >>> print(info)  # {"industry": "银行", "category": "金融服务"}
        
    Note:
        - 支持完整的股票代码格式（包括交易所前缀）
        - 未找到的股票会返回默认分类信息
        - 使用中文行业分类字典进行类别匹配
    """
    # 加载股票行业数据文件
    stock_profile = pd.read_csv(profile_path)

    # 根据股票代码查找对应的行业信息
    profile = stock_profile[stock_profile["stock_id"] == stock_code]
    
    if not profile.empty:
        # 找到股票，获取细分行业名称
        industry = profile["industry"].values[0]
        
        # 根据细分行业匹配主要类别
        for category, industries in ch.items():
            if industry in industries:
                return {"industry": industry, "category": category}
                
        # 如果细分行业不在预定义类别中，归为"其他"
        return {"industry": industry, "category": "其他"}
    else:
        # 股票代码未找到，返回默认分类
        return {"industry": "未知", "category": "其他"}
    

# ============================ 测试示例代码 ============================
# 以下为函数使用示例（已注释）
# stock_code = "SH601728"  # 示例股票代码
# result = get_stock_industry_and_category(stock_code)
# print(result)  # 输出: {"industry": "具体行业", "category": "主要类别"}

import pandas as pd

ch = {
    "制造业": ["家用电器", "半导体", "电气设备", "工程机械", "汽车整车"],
    "能源与资源": ["煤炭开采", "水力发电", "石油加工", "石油开采", "铜"],
    "金融服务": ["银行", "证券", "保险"],
    "消费品": ["白酒", "乳制品", "食品", "中成药"],
    "科技与通信": ["半导体", "软件服务", "电信运营", "新型电力"],
    "交通与运输": ["水运", "船舶"],
    "房地产": ["全国地产"],
    "旅游与服务": ["旅游服务"],
    "化工与制药": ["化工原料", "化学制药"],
    "基础设施与工程": ["建筑工程"]
}

eng = {
    "Manufacturing": ["家用电器", "半导体", "电气设备", "工程机械", "汽车整车", "建筑工程"],
    "Energy and Resources": ["煤炭开采", "水力发电", "石油加工", "石油开采", "铜"],
    "Financial Services": ["银行", "证券", "保险"],
    "Consumer Goods": ["白酒", "乳制品", "食品", "中成药"],
    "Technology and Communication": ["半导体", "软件服务", "电信运营", "新型电力"],
    "Transportation and Logistics": ["水运", "船舶"],
    "Real Estate": ["全国地产"],
    "Tourism and Services": ["旅游服务"],
    "Chemical and Pharmaceuticals": ["化工原料", "化学制药"],
    "Infrastructure and Engineering": ["建筑工程"]
}


def get_stocks_by_industry(industry: str) -> list:
    """
    Get all stock IDs and names for a given industry from CSV file.

    Args:
        industry (str): Industry name to search for

    Returns:
        list[tuple]: List of tuples containing (stock_id, name) pairs

    Raises:
        ValueError: If industry not found in CSV
    """
    try:
        # Read CSV file
        df = pd.read_csv('data/xueqiu_data/stock_profile.csv')

        # Filter by industry and get stock_id, name columns
        filtered_df = df[df['industry'] == industry][['stock_id', 'name']]

        if filtered_df.empty:
            raise ValueError(f"Industry '{industry}' not found in data")

        # Convert to list of tuples
        result = list(zip(filtered_df['stock_id'], filtered_df['name']))

        return result

    except FileNotFoundError:
        raise FileNotFoundError("CSV file 'stock_data.csv' not found")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")


def get_stock_industry_and_category(
    stock_code: str,
    profile_path: str = "/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/data/xueqiu_data/stock_profile.csv",
) -> dict:
    """
    根据股票代码获取行业和类别。

    Args:
        stock_code (str): 股票代码。
        profile_path (str): 股票行业数据文件的路径。

    Returns:
        dict: 包含行业和类别的字典，格式为 {"industry": str, "category": str}。
              如果股票代码未找到，则返回 {"industry": "未知", "category": "其他"}。
    """
    # 加载股票行业数据
    stock_profile = pd.read_csv(profile_path)

    # 查找股票代码对应的行业
    profile = stock_profile[stock_profile["stock_id"] == stock_code]
    if not profile.empty:
        industry = profile["industry"].values[0]
        # 根据行业获取类别
        for category, industries in ch.items():
            if industry in industries:
                return {"industry": industry, "category": category}
        return {"industry": industry, "category": "其他"}
    else:
        # 如果股票代码未找到，返回默认值
        return {"industry": "未知", "category": "其他"}
    

# stock_code = "SH601728"
# result = get_stock_industry_and_category(stock_code)
# print(result)

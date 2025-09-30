# TwinMarket - A股市场模拟系统(1.0版本)



## 📖 项目简介

TwinMarket 是一个创新的股票交易模拟系统，通过集成大语言模型（LLM）技术，模拟真实的股票市场交易环境。系统通过多智能体协作，实现了包括个性化交易策略、社交网络互动、新闻信息分析等在内的全方位市场模拟。

### 🎯 核心特性

- **🤖 智能交易代理**：基于 LLM 的个性化交易决策系统
- **🌐 社交网络模拟**：完整的论坛互动和用户关系网络
- **📊 多维度分析**：整合技术指标、新闻信息、市场情绪等多种因素
- **🎲 行为金融建模**：考虑处置效应、彩票偏好等行为金融因素
- **⚡ 高性能并发**：支持大规模用户并发交易模拟
- **📈 实时撮合引擎**：完整的订单撮合和交易执行系统


## 🚀 快速开始


```bash
# 自行配置api与embedding模型:
cp config/api_example.yaml config/api.yaml
cp config/embedding_example.yaml config/embedding.yaml

# 运行样例
bash script/run.sh
```



## 📝 开发指南

### 扩展交易策略

在 `trader/trading_agent.py` 中实现新的交易策略：

```python
def custom_strategy(self, market_data):
    """自定义交易策略"""
    # 实现你的策略逻辑
    pass
```

### 添加新的评估指标

在 `trader/utility.py` 中添加评估函数：

```python
def calculate_custom_metric(trades):
    """计算自定义指标"""
    # 实现指标计算
    pass
```
# TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets

<p align="center">[ English | <a href="README_zh.md">中文</a> ]</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.01506">
    <img src="https://img.shields.io/badge/arXiv-2502.01506-b31b1b.svg" alt="arXiv:2502.01506" />
  </a>
  
</p>

## 📖 Overview

TwinMarket is an innovative stock market simulation system powered by Large Language Models (LLMs). It simulates realistic trading environments through multi-agent collaboration, covering personalized trading strategies, social network interactions, and news/information analysis for an end-to-end market simulation.

### 🎯 Key Features

- **🤖 Intelligent Trading Agents**: LLM-driven, personalized decision-making
- **🌐 Social Network Simulation**: Forum-style interactions and user relationship graphs
- **📊 Multi-dimensional Analytics**: Technical indicators, news, and market sentiment
- **🎲 Behavioral Finance Modeling**: Includes disposition effect, lottery preference, and more
- **⚡ High-performance Concurrency**: Scalable simulation for large user populations
- **📈 Real-time Matching Engine**: Full order matching and execution

## 🚀 Quick Start

```bash
# Configure your API and embedding models
cp config/api_example.yaml config/api.yaml
cp config/embedding_example.yaml config/embedding.yaml

# Run the demo
bash script/run.sh
```

## 📝 Development Guide

### Extend Trading Strategies

Implement new strategies in `trader/trading_agent.py`:

```python
def custom_strategy(self, market_data):
    """Custom trading strategy"""
    # Implement your strategy logic here
    pass
```

### Add New Evaluation Metrics

Add metrics in `trader/utility.py`:

```python
def calculate_custom_metric(trades):
    """Compute custom metric"""
    # Implement metric calculation here
    pass
```

## 🧾 Citation

```bibtex
@misc{yang2025twinmarketneurips,
      title={TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets},
      author={Yuzhe Yang and Yifei Zhang and Minghao Wu and Kaidi Zhang and
              Yunmiao Zhang and Honghai Yu and Yan Hu and Benyou Wang},
      year={2025},
      eprint={2502.01506},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2502.01506},
}
```
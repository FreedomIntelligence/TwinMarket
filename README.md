# TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets 

<p align="left">

  <a href="https://arxiv.org/abs/2502.01506">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="arXiv:2502.01506" />
  </a>

  <a href="https://freedomintelligence.github.io/TwinMarket/">
    <img src="https://img.shields.io/badge/Project-Page-4caf50.svg" alt="Project Page" />
  </a>
  
  <a href="README.md">
    <img src="https://img.shields.io/badge/🌍-English-blue.svg" alt="English" />
  </a>

  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/🌏-中文-red.svg" alt="中文" />
  </a>

</p>


<div align="center">
  <img src="assets/img/TwinMarket.png" alt="TwinMarket Overview" width="100%" style="max-width: 1000px; margin: 0 auto; display: block;">
</div>

 ## 💡 Update
- **09/2025:** TwinMarket was accepted to NeurIPS 2025. See you in San Diego! 🌊
- **04/2025:** TwinMarket won the [Best Paper Award](https://yuzheyang.com/src/img/best_paper.jpg) 🏆 at the [Advances in Financial AI Workshop @ ICLR 2025](https://sites.google.com/view/financialaiiclr25/home).

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

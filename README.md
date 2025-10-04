![Banner](assets/banner.png)
[![Python](https://img.shields.io/badge/Python-3.12.11-blue?logo=python)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch)](https://pytorch.org/)![Made with ML](https://img.shields.io/badge/Made%20with-ML-blueviolet?logo=openai)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 🤖 Portfolio Optimization with Deep Reinforcement Learning

This project explores the use of Deep Reinforcement Learning to train autonomous agents for financial portfolio management. The goal was not just to create a single profitable agent, but to conduct a comparative study of different RL algorithms (PPO, SAC, TD3) to understand the emergent trading strategies and their robustness across various market conditions.

**The ultimate finding? A TD3-based agent learned a superior, risk-managed static asset allocation that consistently outperformed both active trading strategies and aggressive growth models, especially during market downturns.**

---

## 📜 Table of Contents

1. [📊 The Data & Asset Selection](#-the-data--asset-selection)
2. [🎯 Benchmarking Against Baselines](#-benchmarking-against-baselines)
3. [🏆 Key Findings & The Champion Agent](#-key-findings--the-champion-agent)
4. [🧠 Comparative Analysis of Agent Strategies](#-comparative-analysis-of-agent-strategies)
    * [🥇 TD3: The Prudent Risk-Manager](#-td3-the-prudent-risk-manager)
    * [🚀 SAC: The Aggressive Growth Engine](#-sac-the-aggressive-growth-engine)
    * [📈 PPO: The Active (but Inconsistent) Trader](#-ppo-the-active-but-inconsistent-trader)
5. [🌪️ Stress Testing: The Ultimate Test of Robustness](#️-stress-testing-the-ultimate-test-of-robustness)
6. [🔬 The Research Journey: Why Simplicity Won](#-the-research-journey-why-simplicity-won)
7. [✅ Conclusion](#-conclusion)
8. [📂 Project Structure](#-project-structure)
9. [🚀 How to Run](#-how-to-run)
    * [Setup](#setup)
    * [Data Fetching](#data-fetching)
    * [Training](#training)
    * [Evaluation & Visualization](#evaluation--visualization)

---

## 📊 The Data & Asset Selection

The foundation of any financial machine learning project is the data. This project uses daily closing price data sourced from **Yahoo Finance** via the `yfinance` library. The primary training period was **2015-2020**, with out-of-sample testing conducted on **2021-2023** and other periods for stress testing.

The selection of assets was crucial for creating a realistic decision-making environment for the agent. The portfolio consists of five assets, chosen to represent different classes and risk profiles:

* **Growth Equities (AAPL, MSFT):** Represent the high-growth, high-volatility technology sector.
* **Market Index (SPY):** An ETF tracking the S&P 500, representing the broader US stock market.
* **Safe Haven (TLT):** An ETF for 20+ Year US Treasury Bonds, which often acts as a "risk-off" asset during stock market downturns.
* **Alternative Asset (BTC-USD):** Represents a non-traditional, extremely volatile asset class with high potential returns.

This diverse mix forces the agent to learn not just about individual assets, but also about their correlations and how to balance risk across different economic regimes.

---

## 🎯 Benchmarking Against Baselines

To prove that a reinforcement learning agent is truly "intelligent," its performance must be measured against simple, standard strategies. An agent is only successful if it can provide value beyond a naive approach.

Our primary benchmark was the **Buy and Hold** strategy, where an equal amount of capital is invested in each asset at the beginning of the period and never touched again. The goal for any trained RL agent was to achieve superior performance, especially on a **risk-adjusted basis** (e.g., higher Sharpe Ratio, lower Max Drawdown), compared to this baseline.

The chart below shows the performance of a simple Buy and Hold strategy during the 2021-2023 test period, setting a clear target for our agents to beat.

![Baseline Performance Chart](results/baseline_results.png)

---

## 🏆 Key Findings & The Champion Agent

After extensive training, evaluation, and stress-testing, the **TD3 agent emerged as the clear winner** on a risk-adjusted basis. While other agents achieved higher raw returns, their strategies proved to be brittle and dangerously volatile during market crises. The TD3 agent's strategy was the most robust and reliable.

#### Final Performance Comparison (2021-2023)

This table summarizes the performance of the top-performing static agents against the baseline.

| Metric | **TD3 Agent** | SAC Agent | Buy & Hold |
| :--- | :--- | :--- | :--- |
| **Total Return** | 47.24% | **50.89%** | 34.91% |
| **CAGR** | 13.76% | **14.70%** | 10.50% |
| **Sharpe Ratio** | **0.62** | 0.51 | 0.45 |
| **Max Drawdown** | **-28.41%** | -44.61% | -40.81% |

The TD3 agent delivered strong returns while significantly reducing the maximum drawdown, proving its superior capital preservation strategy.

![Main Performance Chart](results/final_performance_comparison_all_agents.png)

---

## 🧠 Comparative Analysis of Agent Strategies

A fascinating outcome of this project was observing three different RL algorithms independently discover three distinct and recognizable investment philosophies.

### 🥇 TD3: The Prudent Risk-Manager

The TD3 agent concluded that the most effective strategy was not to trade frequently, but to find one **superior, risk-managed static asset allocation** and hold it.

* **Strategy:** "Smarter Buy and Hold".
* **Behavior:** The agent's allocation is completely static, indicating it focused on the initial strategic decision and ignored market noise to minimize transaction costs.
* **Result:** This approach led to the best risk-adjusted returns, proving that a robust initial setup is more valuable than reactive trading.

![TD3 Allocation Chart](results/td3_portfolio_alocation.png)

### 🚀 SAC: The Aggressive Growth Engine

The SAC agent also learned a static allocation strategy, but its portfolio was geared for **maximum growth**, accepting higher risk for higher potential returns.

* **Strategy:** High-risk, high-return static allocation.
* **Behavior:** Like TD3, it made one initial allocation and held firm. However, this allocation was far more aggressive.
* **Result:** It achieved the highest total return in some periods but suffered catastrophic drawdowns in stress tests, making its strategy unreliable and brittle.

![SAC Performance Chart](results/sac_portfolio_alocation.png)

### 📈 PPO: The Active (but Inconsistent) Trader

Unlike the other two, the PPO agent learned an **active, dynamic trading strategy**, constantly adjusting its portfolio based on market conditions.

* **Strategy:** Tactical asset allocation.
* **Behavior:** The allocation chart clearly shows the agent rebalancing its portfolio over time, for example, by increasing its bond (TLT) holdings during the 2022 downturn.
* **Result:** While impressive that it learned this behavior, its performance was inconsistent. It succeeded in some periods (2018) but failed in others (2025), highlighting the immense difficulty of successful market timing.

![PPO Allocation Chart](results/ppo_portfolio_alocation.png)

---

## 🌪️ Stress Testing: The Ultimate Test of Robustness

A model is only as good as its performance during a crisis. We subjected the agents to multiple out-of-sample stress tests, with the 2018 period (featuring a crypto winter and a stock market flash crash) being the most revealing.

![2018 Stress Test Chart](results/stress_test_comparison_2018.png)

* **TD3's Triumph:** The orange line shows the TD3 agent successfully navigating the downturn, preserving capital far better than the baseline.
* **SAC's Failure:** The green line shows the SAC agent's aggressive strategy failing catastrophically, resulting in a massive drawdown.

This test definitively proved that the **TD3 agent's risk-managed approach was truly robust**, while the SAC agent's strategy was fragile.

---

## 🔬 The Research Journey: Why Simplicity Won

This project was also an exercise in scientific methodology. We initially hypothesized that more complex models and features would yield better results.

* **Hypothesis 1: More features are better.** We tested adding technical indicators (RSI, MACD) to the observation space. **Result:** Performance degraded. The indicators acted as noise, confusing the agents.
* **Hypothesis 2: Models with memory are better.** We tested an LSTM-based agent (`RecurrentPPO`). **Result:** Performance degraded. The added complexity led to overfitting on the training data.
* **Hypothesis 3: Using Regularization is better.** We tested both L1 and L2 regularization. **Results:** Performance degraded.
* **Hypothesis 4: Increasing the window from 30 days is better.** We tested increasing the window to 60 days. **Results:** Performance degraded. increasing the context window is not always good and it could be seen as more noise for the model.

The conclusion was clear: for this problem, a simple and elegant model (a standard MLP fed with just normalized price data) was the most effective.

---

## ✅ Conclusion

This project successfully demonstrates that Deep Reinforcement Learning can be a powerful tool for discovering sophisticated investment strategies. The key insight is that the most robust and successful agent did not learn to be a hyperactive trader, but rather a prudent strategic allocator, emphasizing the timeless investment principle that effective risk management is the true key to long-term success.

---

## 📂 Project Structure

The codebase is organized into modular, reusable scripts.

```bash
├── assets/                 
├── checkpoints/            # Holds all saved model .zip files
├── results/                # Holds all output plots and metrics
├── scripts/
│   ├── environment.py      # The custom Gymnasium environment for the simulation
│   ├── fetch_market_data.py# A flexible script to download data for any period
│   ├── train.py            # The main training script with model selection
│   ├── evaluate.py         # The main evaluation script for generating metrics
│   ├── stress_test.py      # Runs a full comparison of all agents on a given dataset
│   └── visualize_strategy.py # Plots the asset allocation of a single trained agent
└── README.md                 # This file
```

---

## 🚀 How to Run

### Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Data Fetching

Use the flexible `fetch_market_data.py` script to get any data you need.

```bash
# Fetch the default training data (2015-2021)
python fetch_market_data.py --start 2015-01-01 --end 2020-12-31 --filename data/train.csv

# Fetch data for a stress test (e.g., 2022)
python fetch_market_data.py --start 2022-01-01 --end 2022-12-31 --filename data/test_2022.csv
```

### Training

Use the `train.py` script to train any of the three main agents.

```bash
# Train the champion TD3 agent (default)
python src/train.py --agent td3

# Train a SAC agent for more timesteps
python src/train.py --agent sac --timesteps 100000
```

### Evaluation & Visualization

Use the dedicated scripts to analyze the results.

```bash
# Run a full stress test on the 2018 data
python stress_test.py --datafile data/stress_test_2018.csv

# Visualize the TD3 agent's strategy
python visualize_strategy.py --agent td3 --checkpoint td3_portfolio_model.zip
```

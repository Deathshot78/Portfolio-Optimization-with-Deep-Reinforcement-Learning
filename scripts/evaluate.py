import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC ,PPO , TD3
from evaluate_baselines import buy_and_hold
from environment import PortfolioEnv
from matplotlib.ticker import FuncFormatter

# --- Helper Function to Run the RL Agent ---

def evaluate_agent(env, model):
    """
    Runs the trained agent on the environment and returns portfolio values.
    """
    obs, info = env.reset()
    terminated, truncated = False, False

    portfolio_values = [env.initial_balance]

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])

    return pd.Series(portfolio_values, index=env.df.index[:len(portfolio_values)])


def calculate_metrics(portfolio_values, freq=252, rf=0.0):
    """
    Calculates key performance metrics from a series of portfolio values.
    freq: number of trading periods in a year (252 for daily, 52 for weekly).
    rf: risk-free rate (default = 0 for simplicity).
    """
    returns = portfolio_values.pct_change().dropna()

    # Total Return
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    # CAGR
    num_years = (len(portfolio_values) / freq)
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/num_years) - 1

    # Sharpe Ratio
    sharpe_ratio = np.sqrt(freq) * (returns.mean() - rf) / returns.std()

    # Sortino Ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = np.sqrt(freq) * (returns.mean() - rf) / downside_std if downside_std > 0 else np.nan

    # Volatility (annualized std)
    volatility = returns.std() * np.sqrt(freq)

    # Max Drawdown
    rolling_max = portfolio_values.cummax()
    drawdown = portfolio_values / rolling_max - 1.0
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown / 100) if max_drawdown != 0 else np.nan

    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Volatility": f"{volatility:.2%}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Calmar Ratio": f"{calmar_ratio:.2f}"
    }


def main(test_data_path='data/test.csv'):
    """
    Loads, evaluates, and plots the performance of PPO, SAC, and TD3 agents
    against a Buy and Hold baseline.
    """
    # --- Define Model Paths and Agent Types ---
    models_to_evaluate = {
        "PPO Agent": (PPO, 'checkpoints/ppo_portfolio_model'),
        "SAC Agent": (SAC, 'checkpoints/sac_portfolio_model'),
        "TD3 Agent": (TD3, 'checkpoints/td3_portfolio_model')
    }

    # Load test data
    test_df = pd.read_csv(test_data_path, index_col='Date', parse_dates=True)

    # Dictionary to store results
    portfolio_values = {}
    metrics = {}

    # --- Run Evaluations for each RL Agent---
    for name, (agent_type, model_path) in models_to_evaluate.items():
        print(f"--- Evaluating {name} ---")
        model = agent_type.load(model_path)
        env = PortfolioEnv(test_df)
        portfolio_values[name] = evaluate_agent(env, model)
        metrics[name] = calculate_metrics(portfolio_values[name])

    # --- Evaluate Buy and Hold Baseline ---
    print("\n--- Evaluating Buy and Hold Baseline ---")
    bnh_values = buy_and_hold(test_df)
    portfolio_values["Buy and Hold"] = bnh_values
    metrics["Buy and Hold"] = calculate_metrics(bnh_values)
    
    # --- Combine and Print Metrics ---
    print("\n--- Performance Metrics ---")
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    # --- Plotting All Strategies ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for clarity
    colors = {
        "PPO Agent": "red",
        "SAC Agent": "green",
        "TD3 Agent": "orange",
        "Buy and Hold": "blue"
    }

    for name, values in portfolio_values.items():
        ax.plot(values.index, values, label=name, color=colors[name], linewidth=2)

    ax.set_title('Agent Performance Comparison', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=12)
    
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig('results/final_performance_comparison_all_agents.png')
    plt.show()

# Example of how to run this main function
if __name__ == '__main__':
    # You can specify a different test file here if needed
    # e.g., main(test_data_path='data/stress_test_2018.csv')
    main()
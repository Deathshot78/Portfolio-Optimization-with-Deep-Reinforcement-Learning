import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from stable_baselines3 import PPO, SAC, TD3
from environment import PortfolioEnv

def visualize_strategy(agent_name, checkpoint_path, datafile_path, output_path):
    """
    Loads a trained agent, runs a simulation, and plots its portfolio allocation strategy.

    Args:
        agent_name (str): The type of agent to load ('ppo', 'sac', 'td3').
        checkpoint_path (str): The path to the saved model checkpoint file (.zip).
        datafile_path (str): The path to the CSV market data for the simulation.
        output_path (str): The path to save the output plot image.
    """
    print(f"--- Visualizing strategy for {agent_name.upper()} agent ---")

    # 1. Define a mapping from agent names to their classes
    AGENT_CLASSES = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3
    }
    agent_class = AGENT_CLASSES[agent_name.lower()]

    # 2. Load Data and Model
    try:
        test_df = pd.read_csv(datafile_path, index_col='Date', parse_dates=True)
        model = agent_class.load(checkpoint_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find a required file. {e}")
        return
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return

    # 3. Create Environment and Run Simulation
    env = PortfolioEnv(test_df)
    obs, info = env.reset()
    terminated, truncated = False, False

    weights_history = [info['weights']]
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        weights_history.append(info['weights'])
    print("✅ Simulation complete.")

    # 4. Prepare Data for Plotting
    weights_df = pd.DataFrame(weights_history)
    asset_names = test_df.columns.tolist() + ['Cash']
    weights_df.columns = asset_names
    weights_df.index = test_df.index[:len(weights_df)]

    # 5. Plotting the Stacked Area Chart
    print("📊 Generating plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns, alpha=0.8)

    ax.set_title(f'Agent Portfolio Allocation Over Time ({agent_name.upper()})', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Allocation (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    formatter = FuncFormatter(lambda y, p: f'{y:.0%}')
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Visualize a trained RL agent's portfolio allocation strategy.")
    
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3"],
        help="The RL algorithm of the trained agent."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint .zip file (e.g., 'td3_portfolio_model.zip')."
    )
    parser.add_argument(
        "--datafile",
        type=str,
        default="data/test.csv",
        help="Path to the market data CSV file to run the simulation on."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/agent_allocation.png",
        help="Path to save the output plot image."
    )

    args = parser.parse_args()

    visualize_strategy(
        agent_name=args.agent,
        checkpoint_path=args.checkpoint,
        datafile_path=args.datafile,
        output_path=args.output
    )
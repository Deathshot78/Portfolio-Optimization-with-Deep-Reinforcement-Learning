import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    A custom reinforcement learning environment for portfolio management.

    This environment simulates the daily trading of multiple financial assets. The agent's
    goal is to learn a policy for allocating capital to maximize risk-adjusted returns.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=30, initial_balance=10000, transaction_cost_pct=0.001):
        """
        Initializes the portfolio management environment.

        Args:
            df (pd.DataFrame): A DataFrame containing the daily closing prices of the assets.
                               The index should be dates and columns should be asset tickers.
            window_size (int): The number of past days of price data to include in the observation.
            initial_balance (float): The starting capital for the portfolio.
            transaction_cost_pct (float): The percentage cost for each trade (e.g., 0.001 for 0.1%).
        """
        super(PortfolioEnv, self).__init__()

        # --- Basic Environment Parameters ---
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.n_assets = len(df.columns)

        # --- Action Space ---
        # The agent outputs a vector of continuous values, one for each asset plus one for cash.
        # These raw outputs are then converted to portfolio weights via a softmax function.
        # The space is defined from -1 to 1 for better compatibility with standard RL algorithms.
        # Shape: (number of assets + 1 for cash)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # --- Observation Space ---
        # The agent observes a window of past price data, flattened into a 1D vector.
        # Shape: (window_size * number of assets)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_assets,),
            dtype=np.float32
        )

        # --- Internal State Variables ---
        # These variables track the state of the simulation over time.
        self._current_step = 0
        self._portfolio_value = 0.0
        # Weights for each asset + cash, e.g., [w_aapl, w_msft, ..., w_cash]
        self._weights = np.zeros(self.n_assets + 1)

    def reset(self, seed=None):
        """
        Resets the environment to its initial state for a new episode.
        
        Returns:
            tuple: A tuple containing the initial observation and auxiliary info.
        """
        super().reset(seed=seed)

        # Start the simulation at the first point where a full window of data is available.
        self._current_step = self.window_size
        self._portfolio_value = self.initial_balance
        
        # Initialize weights to be 100% in cash.
        self._weights = np.zeros(self.n_assets + 1)
        self._weights[-1] = 1.0  # Last element represents cash

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment based on the agent's action.

        Args:
            action (np.ndarray): The raw output from the agent's policy network.

        Returns:
            tuple: A tuple containing the next observation, reward, terminated flag,
                   truncated flag, and auxiliary info.
        """
        # 1. Store the portfolio value before taking the action.
        current_portfolio_value = self._portfolio_value

        # 2. Convert the raw action into portfolio weights using the softmax function.
        # This ensures the weights are positive and sum to 1.
        target_weights = np.exp(action) / np.sum(np.exp(action))

        # 3. Calculate the cost of rebalancing the portfolio.
        # The cost is based on the total value of assets bought or sold.
        trades = (target_weights[:-1] - self._weights[:-1]) * current_portfolio_value
        transaction_costs = np.sum(np.abs(trades)) * self.transaction_cost_pct

        # 4. Update the internal state: apply costs, set new weights, and advance time.
        self._balance = current_portfolio_value - transaction_costs
        self._weights = target_weights
        self._current_step += 1

        # 5. Calculate the new portfolio value based on the market's price movement.
        current_prices = self.df.iloc[self._current_step - 1].values
        next_prices = self.df.iloc[self._current_step].values
        price_ratio = next_prices / current_prices  # How much each asset's price changed.
        
        # The new value of our asset holdings.
        asset_values_after_price_change = (self._weights[:-1] * self._balance) * price_ratio
        
        # The new total portfolio value is the sum of the updated asset values plus the cash holding.
        new_portfolio_value = np.sum(asset_values_after_price_change) + (self._weights[-1] * self._balance)
        self._portfolio_value = new_portfolio_value

        # 6. Calculate the reward for the agent.
        # The reward is the log return of the portfolio value, which encourages geometric growth.
        reward = np.log(new_portfolio_value / current_portfolio_value)

        # 7. Check for termination conditions.
        # The episode ends if the agent goes broke or runs out of data.
        terminated = bool(self._portfolio_value <= self.initial_balance * 0.5)
        truncated = self._current_step >= len(self.df) - 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Constructs the observation for the agent at the current time step.

        Returns:
            np.ndarray: A flattened 1D array of the normalized price history.
        """
        # Get the window of historical price data.
        price_window = self.df.iloc[self._current_step - self.window_size : self._current_step].values
        
        # Normalize the window by dividing by the first price. This helps the agent
        # focus on relative price changes rather than absolute values.
        normalized_window = price_window / price_window[0]
        
        return normalized_window.flatten().astype(np.float32)

    def _get_info(self):
        """
        Returns a dictionary of auxiliary information about the current state.
        """
        return {
            'step': self._current_step,
            'portfolio_value': self._portfolio_value,
            'weights': self._weights,
        }

    def render(self, mode='human'):
        """
        Renders the environment's state (optional).
        """
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, Portfolio Value: {info['portfolio_value']:.2f}")

    def close(self):
        """
        Cleans up the environment (optional).
        """
        pass
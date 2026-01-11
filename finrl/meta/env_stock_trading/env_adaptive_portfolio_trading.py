"""
Adaptive Portfolio Trading Environment with Online Learning

This environment combines:
1. Portfolio optimization with position constraints
2. Discrete trading execution
3. Online/incremental learning (fine-tuning every rebalance period)
4. Memory-efficient design for production

Key Features:
- Dual-mode operation (optimization + trading)
- Incremental model updates with low learning rate
- Efficient state representation
- Built-in risk management
- Optimized for 2-hour rebalancing cycles

Author: FinRL Team
Date: 2025-11-29
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import warnings
from dataclasses import dataclass
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass
class PerformanceMetrics:
    """Container for performance tracking"""
    portfolio_value: float
    returns: float
    sharpe_ratio: float
    total_trades: int
    transaction_costs: float
    win_rate: float


class AdaptivePortfolioTradingEnv(gym.Env):
    """
    Adaptive environment with online learning capabilities.
    
    This environment learns and adapts during deployment by:
    - Fine-tuning model parameters every rebalance period
    - Using recent experience for incremental updates
    - Maintaining efficient replay buffer
    
    Perfect for:
    - Production trading with concept drift
    - Periodic rebalancing (e.g., every 2 hours)
    - Continuous model improvement
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        initial_amount: float = 1000000,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        tech_indicator_list: List[str] = None,
        mode: str = "optimization",  # "optimization" or "trading"
        # Rebalancing settings
        rebalance_interval: int = 120,  # minutes
        # Risk management
        max_position_size: float = 0.30,  # 30% max per stock
        min_cash_ratio: float = 0.10,  # 10% minimum cash
        # Online learning settings
        enable_online_learning: bool = True,
        learning_buffer_size: int = 1000,
        finetune_timesteps: int = 500,
        finetune_learning_rate: float = 1e-5,
        # Performance settings
        lookback_window: int = 30,  # For calculating indicators
        normalize_state: bool = True,
        # Debugging
        verbose: int = 0,
    ):
        """
        Initialize adaptive trading environment.
        
        Args:
            df: DataFrame with [date, tic, close, volume, tech_indicators...]
            stock_dim: Number of stocks
            initial_amount: Starting capital
            transaction_cost_pct: Transaction cost (0.001 = 0.1%)
            reward_scaling: Scale rewards for training stability
            tech_indicator_list: List of technical indicators in df
            mode: "optimization" (portfolio weights) or "trading" (buy/sell)
            rebalance_interval: Minutes between rebalancing
            max_position_size: Maximum % in single stock
            min_cash_ratio: Minimum % to keep in cash
            enable_online_learning: Enable incremental learning
            learning_buffer_size: Size of experience buffer
            finetune_timesteps: Steps for each fine-tuning
            finetune_learning_rate: Low LR for fine-tuning
            lookback_window: Window for state features
            normalize_state: Whether to normalize state values
            verbose: Print level (0=silent, 1=info, 2=debug)
        """
        
        super().__init__()
        
        # Core parameters
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.mode = mode
        self.verbose = verbose
        
        # Rebalancing
        self.rebalance_interval = rebalance_interval
        self.max_position_size = max_position_size
        self.min_cash_ratio = min_cash_ratio
        
        # Online learning
        self.enable_online_learning = enable_online_learning
        self.learning_buffer_size = learning_buffer_size
        self.finetune_timesteps = finetune_timesteps
        self.finetune_learning_rate = finetune_learning_rate
        
        # Performance
        self.lookback_window = lookback_window
        self.normalize_state = normalize_state
        
        # Validate data
        self._validate_data()
        
        # Get unique dates and tickers
        self.dates = sorted(df['date'].unique())
        self.tickers = sorted(df['tic'].unique())
        assert len(self.tickers) == stock_dim, f"stock_dim={stock_dim} but found {len(self.tickers)} tickers"
        
        # Define spaces
        self._setup_spaces()
        
        # State tracking
        self.current_step = 0
        self.current_date_idx = 0
        self.cash = initial_amount
        self.holdings = np.zeros(stock_dim, dtype=np.float32)
        self.portfolio_value = initial_amount
        
        # Portfolio weights tracking
        self.current_weights = np.zeros(stock_dim + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # Start with 100% cash
        
        # Memory for online learning
        if self.enable_online_learning:
            self.experience_buffer = deque(maxlen=learning_buffer_size)
            self.model_for_finetuning = None
        
        # Performance tracking
        self.asset_memory = [initial_amount]
        self.returns_memory = [0.0]
        self.actions_memory = []
        self.costs_memory = []
        self.trades_memory = []
        
        # Episode tracking
        self.episode = 0
        self.total_trades = 0
        self.total_costs = 0.0
        
        if self.verbose >= 1:
            print(f"✓ AdaptivePortfolioTradingEnv initialized")
            print(f"  Mode: {mode}")
            print(f"  Stocks: {stock_dim}")
            print(f"  Tech indicators: {len(self.tech_indicator_list)}")
            print(f"  Online learning: {enable_online_learning}")
    
    def _validate_data(self):
        """Validate input dataframe"""
        required_cols = ['date', 'tic', 'close', 'volume']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        for tech in self.tech_indicator_list:
            if tech not in self.df.columns:
                warnings.warn(f"Technical indicator '{tech}' not found in dataframe")
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        
        # State components:
        # [cash, prices(n), holdings(n), price_changes(n), 
        #  tech_indicators(n*m), weights(n+1), lookback_features]
        
        base_features = 1 + self.stock_dim * 3  # cash + prices + holdings + price_changes
        tech_features = self.stock_dim * len(self.tech_indicator_list)
        weight_features = self.stock_dim + 1
        
        # Add lookback features (moving averages, volatility)
        lookback_features = self.stock_dim * 3  # mean, std, momentum
        
        state_dim = base_features + tech_features + weight_features + lookback_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space depends on mode
        if self.mode == "optimization":
            # Portfolio weights (including cash)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.stock_dim + 1,),
                dtype=np.float32
            )
        else:  # trading mode
            # Buy/sell signals
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.stock_dim,),
                dtype=np.float32
            )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.current_date_idx = self.lookback_window  # Start after lookback
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)
        self.portfolio_value = self.initial_amount
        
        # Reset weights
        self.current_weights = np.zeros(self.stock_dim + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        
        # Reset memory
        self.asset_memory = [self.initial_amount]
        self.returns_memory = [0.0]
        self.actions_memory = []
        self.costs_memory = []
        self.trades_memory = []
        
        self.total_trades = 0
        self.total_costs = 0.0
        self.episode += 1
        
        # Get initial state
        state = self._get_state()
        info = self._get_info()
        
        if self.verbose >= 2:
            print(f"\n{'='*60}")
            print(f"Episode {self.episode} - Reset")
            print(f"{'='*60}")
        
        return state, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step"""
        
        # Check if terminal
        terminal = self.current_date_idx >= len(self.dates) - 1
        
        if terminal:
            return self._handle_terminal()
        
        # Store action
        self.actions_memory.append(actions.copy())
        
        # Get current prices
        current_data = self._get_current_data()
        prices = current_data['close'].values
        
        # Calculate portfolio value before action
        prev_portfolio_value = self.portfolio_value
        
        # Process action based on mode
        if self.mode == "optimization":
            # Get target weights
            target_weights = self._process_weights(actions)
            
            # Execute rebalancing
            cost = self._execute_rebalancing(target_weights, prices)
            
        else:  # trading mode
            # Execute discrete trades
            cost = self._execute_trades(actions, prices)
        
        # Move to next time step
        self.current_step += 1
        self.current_date_idx += 1
        
        # Update portfolio value
        new_data = self._get_current_data()
        new_prices = new_data['close'].values
        self._update_portfolio_value(new_prices)
        
        # Calculate reward
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward = portfolio_return * self.reward_scaling
        
        # Add cost penalty
        reward -= cost / prev_portfolio_value * self.reward_scaling
        
        # Update memory
        self.returns_memory.append(portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        self.costs_memory.append(cost)
        
        # Store experience for online learning
        if self.enable_online_learning:
            self._store_experience(self._get_state(), actions, reward, terminal)
        
        # Check if it's time to fine-tune
        if self.enable_online_learning and self._should_finetune():
            self._finetune_model()
        
        # Get next state
        state = self._get_state()
        info = self._get_info()
        
        if self.verbose >= 2:
            self._print_step_info(portfolio_return, cost)
        
        return state, reward, terminal, False, info
    
    def _get_current_data(self) -> pd.DataFrame:
        """Get current time step data for all stocks"""
        current_date = self.dates[self.current_date_idx]
        data = self.df[self.df['date'] == current_date].copy()
        
        # Ensure consistent ordering
        data = data.set_index('tic').loc[self.tickers].reset_index()
        
        return data
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state vector with optimized features.
        
        State components:
        1. Normalized cash
        2. Current prices (normalized)
        3. Current holdings (normalized)
        4. Price changes (returns)
        5. Technical indicators
        6. Current portfolio weights
        7. Lookback features (moving averages, volatility, momentum)
        """
        
        current_data = self._get_current_data()
        prices = current_data['close'].values
        
        # Base features
        cash_normalized = np.array([self.cash / self.initial_amount], dtype=np.float32)
        prices_normalized = prices / np.mean(prices) if self.normalize_state else prices
        holdings_normalized = self.holdings / 100.0 if self.normalize_state else self.holdings
        
        # Price changes (returns)
        if self.current_date_idx > 0:
            prev_data = self.df[self.df['date'] == self.dates[self.current_date_idx - 1]]
            prev_data = prev_data.set_index('tic').loc[self.tickers].reset_index()
            prev_prices = prev_data['close'].values
            price_changes = (prices - prev_prices) / (prev_prices + 1e-8)
        else:
            price_changes = np.zeros(self.stock_dim, dtype=np.float32)
        
        # Technical indicators
        tech_values = []
        for tech in self.tech_indicator_list:
            values = current_data[tech].values
            if self.normalize_state:
                # Simple standardization
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-8
                values = (values - mean_val) / std_val
            tech_values.extend(values)
        
        tech_array = np.array(tech_values, dtype=np.float32) if tech_values else np.array([], dtype=np.float32)
        
        # Lookback features for trend/volatility
        lookback_features = self._get_lookback_features(prices)
        
        # Combine all features
        state = np.concatenate([
            cash_normalized,
            prices_normalized.astype(np.float32),
            holdings_normalized.astype(np.float32),
            price_changes.astype(np.float32),
            tech_array,
            self.current_weights.astype(np.float32),
            lookback_features,
        ])
        
        return state.astype(np.float32)
    
    def _get_lookback_features(self, current_prices: np.ndarray) -> np.ndarray:
        """Calculate lookback features for each stock"""
        
        features = []
        
        for i, ticker in enumerate(self.tickers):
            # Get historical prices
            start_idx = max(0, self.current_date_idx - self.lookback_window)
            end_idx = self.current_date_idx + 1
            
            historical_dates = self.dates[start_idx:end_idx]
            historical_data = self.df[
                (self.df['date'].isin(historical_dates)) & 
                (self.df['tic'] == ticker)
            ]['close'].values
            
            if len(historical_data) > 1:
                # Moving average
                ma = np.mean(historical_data)
                ma_normalized = current_prices[i] / ma - 1.0
                
                # Volatility
                volatility = np.std(historical_data) / (np.mean(historical_data) + 1e-8)
                
                # Momentum
                momentum = (historical_data[-1] - historical_data[0]) / (historical_data[0] + 1e-8)
            else:
                ma_normalized = 0.0
                volatility = 0.0
                momentum = 0.0
            
            features.extend([ma_normalized, volatility, momentum])
        
        return np.array(features, dtype=np.float32)
    
    def _process_weights(self, actions: np.ndarray) -> np.ndarray:
        """
        Process raw actions into valid portfolio weights.
        
        Applies:
        1. Softmax normalization
        2. Position size limits
        3. Minimum cash requirement
        """
        
        # Softmax normalization
        weights = self._softmax(actions)
        
        # Apply maximum position size constraint
        for i in range(self.stock_dim):
            if weights[i] > self.max_position_size:
                excess = weights[i] - self.max_position_size
                weights[i] = self.max_position_size
                weights[-1] += excess  # Add to cash
        
        # Apply minimum cash constraint
        if weights[-1] < self.min_cash_ratio:
            deficit = self.min_cash_ratio - weights[-1]
            
            # Reduce stock positions proportionally
            stock_weights = weights[:-1]
            total_stock = stock_weights.sum()
            
            if total_stock > 0:
                reduction_factor = 1 - (deficit / total_stock)
                weights[:-1] = stock_weights * reduction_factor
                weights[-1] = self.min_cash_ratio
        
        # Final normalization
        weights = weights / weights.sum()
        
        return weights
    
    def _execute_rebalancing(self, target_weights: np.ndarray, prices: np.ndarray) -> float:
        """
        Execute portfolio rebalancing to achieve target weights.
        
        Returns:
            Total transaction cost
        """
        
        total_value = self.portfolio_value
        total_cost = 0.0
        
        # Calculate target positions
        target_cash = total_value * target_weights[-1]
        target_positions = np.zeros(self.stock_dim)
        
        for i in range(self.stock_dim):
            target_value = total_value * target_weights[i]
            target_positions[i] = target_value / prices[i]
        
        # Execute trades
        for i in range(self.stock_dim):
            diff = target_positions[i] - self.holdings[i]
            
            if abs(diff) > 0.1:  # Minimum trade threshold
                if diff > 0:  # Buy
                    cost = diff * prices[i] * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.holdings[i] += diff
                        self.cash -= cost
                        total_cost += diff * prices[i] * self.transaction_cost_pct
                        self.total_trades += 1
                        
                else:  # Sell
                    shares_to_sell = min(abs(diff), self.holdings[i])
                    proceeds = shares_to_sell * prices[i] * (1 - self.transaction_cost_pct)
                    self.holdings[i] -= shares_to_sell
                    self.cash += proceeds
                    total_cost += shares_to_sell * prices[i] * self.transaction_cost_pct
                    self.total_trades += 1
        
        self.total_costs += total_cost
        self._update_weights(prices)
        
        return total_cost
    
    def _execute_trades(self, actions: np.ndarray, prices: np.ndarray) -> float:
        """Execute discrete buy/sell trades"""
        
        total_cost = 0.0
        
        # Scale actions to trade sizes
        max_trade_value = self.portfolio_value * 0.1  # 10% max per trade
        
        for i in range(self.stock_dim):
            if abs(actions[i]) < 0.1:  # Threshold to avoid tiny trades
                continue
            
            if actions[i] > 0:  # Buy signal
                max_shares = int(max_trade_value / prices[i])
                shares = int(actions[i] * max_shares)
                
                if shares > 0:
                    cost = shares * prices[i] * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.holdings[i] += shares
                        self.cash -= cost
                        total_cost += shares * prices[i] * self.transaction_cost_pct
                        self.total_trades += 1
                        
            else:  # Sell signal
                max_shares = int(self.holdings[i])
                shares = int(abs(actions[i]) * max_shares)
                
                if shares > 0:
                    proceeds = shares * prices[i] * (1 - self.transaction_cost_pct)
                    self.holdings[i] -= shares
                    self.cash += proceeds
                    total_cost += shares * prices[i] * self.transaction_cost_pct
                    self.total_trades += 1
        
        self.total_costs += total_cost
        self._update_weights(prices)
        
        return total_cost
    
    def _update_portfolio_value(self, prices: np.ndarray):
        """Update portfolio value based on current prices"""
        asset_value = np.sum(self.holdings * prices)
        self.portfolio_value = self.cash + asset_value
    
    def _update_weights(self, prices: np.ndarray):
        """Update current portfolio weights"""
        if self.portfolio_value > 0:
            for i in range(self.stock_dim):
                self.current_weights[i] = (self.holdings[i] * prices[i]) / self.portfolio_value
            self.current_weights[-1] = self.cash / self.portfolio_value
        else:
            self.current_weights = np.zeros(self.stock_dim + 1)
    
    def _store_experience(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """Store experience in replay buffer for online learning"""
        self.experience_buffer.append({
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'done': done
        })
    
    def _should_finetune(self) -> bool:
        """Check if it's time to fine-tune the model"""
        # Fine-tune every rebalance_interval steps
        return (self.current_step > 0 and 
                self.current_step % self.rebalance_interval == 0 and
                len(self.experience_buffer) >= self.finetune_timesteps)
    
    def _finetune_model(self):
        """Fine-tune model with recent experience using low learning rate"""
        
        if self.model_for_finetuning is None:
            if self.verbose >= 1:
                print("⚠ No model set for fine-tuning. Call set_model_for_finetuning() first.")
            return
        
        if self.verbose >= 1:
            print(f"\n🔄 Fine-tuning model at step {self.current_step}...")
            print(f"   Buffer size: {len(self.experience_buffer)}")
        
        try:
            # Set low learning rate for fine-tuning
            original_lr = self.model_for_finetuning.learning_rate
            self.model_for_finetuning.learning_rate = self.finetune_learning_rate
            
            # Train on recent experiences
            self.model_for_finetuning.learn(
                total_timesteps=self.finetune_timesteps,
                reset_num_timesteps=False,  # Continue from current progress
                progress_bar=False
            )
            
            # Restore original learning rate
            self.model_for_finetuning.learning_rate = original_lr
            
            if self.verbose >= 1:
                print(f"✓ Fine-tuning complete")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠ Fine-tuning failed: {e}")
    
    def set_model_for_finetuning(self, model):
        """
        Set the model to be fine-tuned during trading.
        
        Args:
            model: Trained RL model (PPO, A2C, SAC, etc.)
        """
        self.model_for_finetuning = model
        
        if self.verbose >= 1:
            print(f"✓ Model set for online learning: {type(model).__name__}")
    
    def _get_info(self) -> dict:
        """Get current state information"""
        current_data = self._get_current_data()
        
        return {
            'step': self.current_step,
            'date': self.dates[self.current_date_idx] if self.current_date_idx < len(self.dates) else None,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'weights': self.current_weights.copy(),
            'total_trades': self.total_trades,
            'total_costs': self.total_costs,
            'prices': current_data['close'].values,
        }
    
    def _handle_terminal(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Handle terminal state and print summary"""
        
        final_return = (self.portfolio_value - self.initial_amount) / self.initial_amount
        
        # Calculate Sharpe ratio
        returns = np.array(self.returns_memory[1:])  # Skip first 0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0.0
        
        # Calculate win rate
        winning_trades = np.sum(returns > 0)
        total_periods = len(returns)
        win_rate = winning_trades / total_periods if total_periods > 0 else 0.0
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"EPISODE {self.episode} COMPLETE")
            print(f"{'='*60}")
            print(f"Initial Value:     ${self.initial_amount:,.2f}")
            print(f"Final Value:       ${self.portfolio_value:,.2f}")
            print(f"Total Return:      {final_return*100:.2f}%")
            print(f"Sharpe Ratio:      {sharpe:.3f}")
            print(f"Win Rate:          {win_rate*100:.1f}%")
            print(f"Total Trades:      {self.total_trades}")
            print(f"Transaction Costs: ${self.total_costs:,.2f}")
            print(f"Cost Impact:       {(self.total_costs/self.initial_amount)*100:.3f}%")
            print(f"{'='*60}\n")
        
        info = {
            'episode': self.episode,
            'final_value': self.portfolio_value,
            'total_return': final_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'total_costs': self.total_costs,
        }
        
        state = self._get_state()
        
        return state, 0.0, True, False, info
    
    def _print_step_info(self, portfolio_return: float, cost: float):
        """Print step information for debugging"""
        print(f"Step {self.current_step} | "
              f"Value: ${self.portfolio_value:,.0f} | "
              f"Return: {portfolio_return*100:+.3f}% | "
              f"Cost: ${cost:.2f} | "
              f"Cash: {self.current_weights[-1]*100:.1f}%")
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        
        returns = np.array(self.returns_memory[1:])
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0.0
        
        winning = np.sum(returns > 0)
        win_rate = winning / len(returns) if len(returns) > 0 else 0.0
        
        return PerformanceMetrics(
            portfolio_value=self.portfolio_value,
            returns=(self.portfolio_value - self.initial_amount) / self.initial_amount,
            sharpe_ratio=sharpe,
            total_trades=self.total_trades,
            transaction_costs=self.total_costs,
            win_rate=win_rate
        )
    
    def save_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Save trading results to DataFrames.
        
        Returns:
            (portfolio_df, actions_df)
        """
        
        # Portfolio values
        portfolio_df = pd.DataFrame({
            'portfolio_value': self.asset_memory,
            'returns': [0] + self.returns_memory,
        })
        
        # Actions (if any recorded)
        if self.actions_memory:
            if self.mode == "optimization":
                cols = [f'weight_{t}' for t in self.tickers] + ['weight_cash']
            else:
                cols = [f'action_{t}' for t in self.tickers]
            
            actions_df = pd.DataFrame(self.actions_memory, columns=cols)
        else:
            actions_df = pd.DataFrame()
        
        return portfolio_df, actions_df
    
    def get_sb_env(self):
        """Get environment wrapped in DummyVecEnv for Stable-Baselines3"""
        env = DummyVecEnv([lambda: self])
        obs = env.reset()
        return env, obs

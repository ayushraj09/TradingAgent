"""
Trading utilities module for paper trading system.
Includes helper functions and fine-tuning logic.
"""
import numpy as np
import pandas as pd
import threading
import tempfile
import os
from datetime import datetime
from pathlib import Path
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def sigmoid_sign(ary, thresh):
    """Sigmoid transformation for turbulence."""
    def sigmoid(x):
        return 1 / (1 + np.exp(-x * np.e)) - 0.5
    return sigmoid(ary / thresh) * thresh


def submit_order(alpaca, qty, stock, side, resp):
    """Submit order to Alpaca."""
    if qty > 0:
        try:
            alpaca.submit_order(stock, qty, side, "market", "day")
            print(f"    ✓ {side.upper()} {qty} {stock}")
            resp.append(True)
        except Exception as e:
            print(f"    ✗ {side.upper()} {qty} {stock} failed: {e}")
            resp.append(False)
    else:
        resp.append(True)


def get_state_from_alpaca(alpaca_processor, config=None):
    """
    Get current state from Alpaca (Production format: 301 features).
    
    State vector: [cash(1)] + [prices(30)] + [stocks(30)] + [tech_indicators(240)]
    NO turbulence in state vector!
    
    Returns:
        state, price, stocks, cash, turbulence, turbulence_bool, tech
    """
    # Load config if not provided
    if config is None:
        from config import load_config
        config = load_config(enable_explanations=False)
    
    # Fetch latest data with technical indicators
    price, tech, turbulence = alpaca_processor.fetch_latest_data(
        ticker_list=config['TICKERS'],
        time_interval='1Min',
        tech_indicator_list=config['TECH_INDICATORS']
    )
    
    # Determine turbulence threshold
    turbulence_bool = 1 if turbulence >= config['TURBULENCE_THRESHOLD'] else 0
    
    # Scale tech indicators
    tech_scaled = tech * 2 ** -7
    
    # Get current positions from Alpaca
    alpaca = tradeapi.REST(
        config['API_KEY'],
        config['API_SECRET'],
        config['API_BASE_URL'],
        'v2'
    )
    
    positions = alpaca.list_positions()
    stocks = np.zeros(config['STOCK_DIM'])
    for position in positions:
        if position.symbol in config['TICKERS']:
            ind = config['TICKERS'].index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))
    
    # Get current cash
    cash = float(alpaca.get_account().cash)
    
    # Build state vector (NO turbulence!)
    # Model expects: 1 (cash) + 30 (prices) + 30 (stocks) + 240 (tech) = 301 features
    amount = np.array(cash * (2 ** -12), dtype=np.float32)
    scale = np.array(2 ** -6, dtype=np.float32)
    
    state = np.hstack((
        amount,
        price * scale,
        stocks * scale,
        tech_scaled,
    )).astype(np.float32)
    
    # Handle NaN/Inf
    state[np.isnan(state)] = 0.0
    state[np.isinf(state)] = 0.0
    
    return state, price, stocks, cash, turbulence, turbulence_bool, tech


def create_env(df, config):
    """Create StockTradingEnv for training/evaluation."""
    state_space = 1 + 2 * config['STOCK_DIM'] + len(config['TECH_INDICATORS']) * config['STOCK_DIM']
    
    df_indexed = df.copy()
    
    # Validate critical columns (price data)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_indexed.columns:
            # Check for zero or negative prices
            invalid_prices = (df_indexed[col] <= 0) | df_indexed[col].isna()
            if invalid_prices.any():
                print(f"⚠️  Warning: Found {invalid_prices.sum()} invalid prices in '{col}', forward filling...")
                # Forward fill from last valid value
                df_indexed[col] = df_indexed.groupby('tic')[col].transform(
                    lambda x: x.replace(0, np.nan).ffill().bfill()
                )
                # If still zero, use mean price for that ticker
                still_invalid = (df_indexed[col] <= 0) | df_indexed[col].isna()
                if still_invalid.any():
                    ticker_means = df_indexed.groupby('tic')[col].transform(
                        lambda x: x[x > 0].mean() if (x > 0).any() else 100.0
                    )
                    df_indexed.loc[still_invalid, col] = ticker_means[still_invalid]
    
    # Clean NaN/Inf values in other columns
    df_indexed = df_indexed.replace([np.inf, -np.inf], np.nan)
    df_indexed = df_indexed.ffill().bfill()
    df_indexed = df_indexed.fillna(0)
    
    # Final validation - ensure no zeros in price columns
    for col in price_cols:
        if col in df_indexed.columns:
            if (df_indexed[col] <= 0).any():
                print(f"✗ Critical: Still have zero prices in '{col}' after cleaning")
                # Use a default price as last resort
                df_indexed.loc[df_indexed[col] <= 0, col] = 100.0
    
    df_indexed = df_indexed.sort_values(['day', 'tic'])
    df_indexed = df_indexed.set_index('day')
    
    env = StockTradingEnv(
        df=df_indexed,
        stock_dim=config['STOCK_DIM'],
        hmax=config['HMAX'],
        initial_amount=config['INITIAL_CASH'],
        num_stock_shares=[0] * config['STOCK_DIM'],
        buy_cost_pct=[config['TRANSACTION_COST_PCT']] * config['STOCK_DIM'],
        sell_cost_pct=[config['TRANSACTION_COST_PCT']] * config['STOCK_DIM'],
        reward_scaling=config['REWARD_SCALING'],
        state_space=state_space,
        action_space=config['STOCK_DIM'],
        tech_indicator_list=config['TECH_INDICATORS'],
        print_verbosity=100000,
    )
    
    return DummyVecEnv([lambda: env])


def evaluate_model_on_df(model, df, config):
    """Evaluate model performance on DataFrame."""
    env = create_env(df, config)
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    
    return total_reward


def finetune_model_with_validation(model, csv_path, config):
    """
    Fine-tune model using recent CSV data with validation.
    Returns: (model, result_dict)
    """
    from data_manager import load_recent_data_from_csv
    
    print(f"\n{'='*80}")
    print("FINE-TUNING MODEL")
    print(f"{'='*80}")
    
    # Load data
    df = load_recent_data_from_csv(csv_path, config, hours=config['FINETUNE_LOOKBACK_HOURS'])
    
    if df is None or len(df) < 500:  # Need substantial data for quality fine-tuning
        print(f"✗ Insufficient data for fine-tuning (have {len(df) if df is not None else 0} rows, need 500+)")
        print(f"   Continue trading - will fine-tune when more data is available")
        return model, None
    
    # Check for NaN/Inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().any().any():
        print("⚠️  Cleaning NaN values in data...")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Critical: Check for zero prices (would cause division by zero)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            zero_prices = (df[col] == 0).sum()
            if zero_prices > 0:
                print(f"⚠️  Found {zero_prices} zero prices in '{col}' - data quality issue")
                # Check if it's too much bad data
                if zero_prices > len(df) * 0.1:  # More than 10% bad data
                    print(f"✗ Too much bad data ({zero_prices}/{len(df)} = {zero_prices/len(df)*100:.1f}%)")
                    print(f"   Skipping fine-tuning - collect more quality data first")
                    return model, None
    
    # Get unique dates
    unique_dates = sorted(df['date'].unique())
    
    # Need at least 20 timestamps for meaningful split (4 for validation)
    if len(unique_dates) < 20:
        print(f"✗ Need at least 20 timestamps for fine-tuning (have {len(unique_dates)})")
        print(f"   Collect more data before fine-tuning")
        return model, None
    
    # Split train/validation
    split_idx = max(len(unique_dates) - 2, int(len(unique_dates) * (1 - config['VALIDATION_SPLIT'])))
    
    train_df = df[df['date'].isin(unique_dates[:split_idx])].copy()
    val_df = df[df['date'].isin(unique_dates[split_idx:])].copy()
    
    # Reset day indices
    train_dates = sorted(train_df['date'].unique())
    val_dates = sorted(val_df['date'].unique())
    
    train_date_to_day = {date: idx for idx, date in enumerate(train_dates)}
    val_date_to_day = {date: idx for idx, date in enumerate(val_dates)}
    
    train_df['day'] = train_df['date'].map(train_date_to_day)
    val_df['day'] = val_df['date'].map(val_date_to_day)
    
    print(f"  Train: {len(train_df):,} rows ({len(train_dates)} timestamps)")
    print(f"  Val: {len(val_df):,} rows ({len(val_dates)} timestamps)")
    
    # Evaluate original
    print(f"  Evaluating original model...")
    try:
        original_score = evaluate_model_on_df(model, val_df, config)
        print(f"  Original score: {original_score:.2f}")
    except (ValueError, RuntimeError) as e:
        if 'nan' in str(e).lower():
            print(f"✗ Data contains NaN/Inf values - skipping fine-tuning")
            print(f"   Error: {str(e)[:100]}...")
            return model, None
        else:
            raise
    
    # Clone model
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name
        model.save(tmp_path)
        model_ft = PPO.load(tmp_path)
    os.remove(tmp_path)
    
    # Fine-tune
    print(f"  Fine-tuning ({config['FINETUNE_STEPS']} steps, lr={config['FINETUNE_LR']})...")
    model_ft.learning_rate = config['FINETUNE_LR']
    ft_env = create_env(train_df, config)
    model_ft.set_env(ft_env)
    model_ft.learn(
        total_timesteps=config['FINETUNE_STEPS'],
        reset_num_timesteps=False,
        progress_bar=False
    )
    
    # Evaluate fine-tuned
    print(f"  Evaluating fine-tuned model...")
    finetuned_score = evaluate_model_on_df(model_ft, val_df, config)
    print(f"  Fine-tuned score: {finetuned_score:.2f}")
    
    # Decision
    threshold = original_score * config['ROLLBACK_THRESHOLD']
    accepted = finetuned_score >= threshold
    improvement = ((finetuned_score - original_score) / original_score * 100) if original_score != 0 else 0
    
    result = {
        'timestamp': datetime.utcnow(),
        'original_score': original_score,
        'finetuned_score': finetuned_score,
        'threshold': threshold,
        'accepted': accepted,
        'improvement_pct': improvement,
        'train_records': len(train_df),
        'val_records': len(val_df),
    }
    
    if accepted:
        print(f"✅ ACCEPTED (+{improvement:.2f}%)")
        return model_ft, result
    else:
        print(f"❌ REJECTED ({improvement:.2f}%)")
        return model, result

"""
Fine-Tuning Simulation Script

Runs production trading simulation with fine-tuning every 2 hours.
- Uses pre-trained model from trained_models/agent_ppo.zip
- Simulates trading on trade_data.csv (3 months of data)
- Fine-tunes with last 48h of train_data + current session
- Validates and accepts/rejects fine-tuned model

Usage:
    python finetune_simulation.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import copy
import warnings
warnings.filterwarnings('ignore')

# Add FinRL to path
sys.path.append(str(Path(__file__).parent / 'FinRL'))

from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# ==================== CONFIGURATION ====================

CONFIG = {
    # Paths
    'TRAINED_MODEL': 'FinRL/examples/trained_models/agent_ppo.zip',
    'TRAIN_DATA': 'FinRL/examples/train_data.csv',
    'TRADE_DATA': 'FinRL/examples/trade_data.csv',
    'OUTPUT_DIR': 'FinRL/examples/fine_tuning_results',
    
    # Trading
    'TRADE_INTERVAL_HOURS': 2,
    'INITIAL_CASH': 1_000_000,
    'STOCK_DIM': 30,
    'HMAX': 10000,
    
    # Fine-tuning
    'FINETUNE_LR': 1e-5,
    'FINETUNE_STEPS': 2000,
    'HISTORICAL_WINDOW_HOURS': 48,
    'VALIDATION_SPLIT': 0.2,
    'ROLLBACK_THRESHOLD': 0.95,
    
    # Environment
    'TRANSACTION_COST_PCT': 0.001,
    'REWARD_SCALING': 1e-4,
}

TECH_INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']

# ==================== HELPER FUNCTIONS ====================

def load_data():
    """Load and preprocess data."""
    print("\n📂 Loading data...")
    
    train_df = pd.read_csv(CONFIG['TRAIN_DATA'])
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    train_df['date'] = pd.to_datetime(train_df['date'])
    if 'vixy' in train_df.columns:
        train_df = train_df.drop(columns=['vixy'])
    
    trade_df = pd.read_csv(CONFIG['TRADE_DATA'])
    if 'Unnamed: 0' in trade_df.columns:
        trade_df = trade_df.drop(columns=['Unnamed: 0'])
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    if 'vixy' in trade_df.columns:
        trade_df = trade_df.drop(columns=['vixy'])
    
    # CRITICAL: StockTradingEnv needs DataFrame indexed by unique timestamps
    # Group by date and assign integer index for each unique date
    train_df = train_df.sort_values(['date', 'tic']).reset_index(drop=True)
    trade_df = trade_df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Create a mapping from unique dates to day index
    train_unique_dates = sorted(train_df['date'].unique())
    trade_unique_dates = sorted(trade_df['date'].unique())
    
    train_date_to_idx = {date: idx for idx, date in enumerate(train_unique_dates)}
    trade_date_to_idx = {date: idx for idx, date in enumerate(trade_unique_dates)}
    
    # Add day index column that StockTradingEnv will use
    train_df['day'] = train_df['date'].map(train_date_to_idx)
    trade_df['day'] = trade_df['date'].map(trade_date_to_idx)
    
    # Set index to day for StockTradingEnv compatibility
    train_df = train_df.set_index('day', drop=False)
    trade_df = trade_df.set_index('day', drop=False)
    
    print(f"✓ Train: {len(train_df):,} rows, {len(train_unique_dates)} days")
    print(f"  Period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"✓ Trade: {len(trade_df):,} rows, {len(trade_unique_dates)} days") 
    print(f"  Period: {trade_df['date'].min()} to {trade_df['date'].max()}")
    
    return train_df, trade_df


def get_trading_timestamps(trade_df, interval_hours=2):
    """Generate trading timestamps every N hours."""
    all_times = sorted(trade_df['date'].unique())
    
    timestamps_by_day = {}
    for ts in all_times:
        day = ts.date()
        if day not in timestamps_by_day:
            timestamps_by_day[day] = []
        timestamps_by_day[day].append(ts)
    
    trading_times = []
    for day, day_times in timestamps_by_day.items():
        day_start = day_times[0]
        trading_times.append(day_start)
        
        current_time = day_start
        for ts in day_times:
            if (ts - current_time).total_seconds() >= interval_hours * 3600:
                trading_times.append(ts)
                current_time = ts
    
    print(f"✓ Trading timestamps: {len(trading_times)} (every {interval_hours}h)")
    return trading_times


def create_env(df, config):
    """Create StockTradingEnv with properly indexed data."""
    state_space = 1 + 2 * config['STOCK_DIM'] + len(TECH_INDICATORS) * config['STOCK_DIM']
    
    # StockTradingEnv expects:
    # - Simple integer index (NOT multi-index)
    # - df.loc[day, :] returns all rows for that day
    # - 'tic' column must exist
    # - 'date' column must exist
    
    # The df already has 'day' column from load_data()
    # Just make sure it's sorted and has the right index
    df_indexed = df.copy()
    df_indexed = df_indexed.sort_values(['day', 'tic'])
    
    # Use 'day' column as the index (NOT multi-index!)
    # This allows df.loc[0, :] to return all tickers for day 0
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
        tech_indicator_list=TECH_INDICATORS,
        print_verbosity=100000,
    )
    
    return DummyVecEnv([lambda: env])


def evaluate_model(model, df, config):
    """Evaluate model on data and return cumulative reward."""
    env = create_env(df, config)
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    
    return total_reward


def finetune_model(model, train_df, trade_df, current_time, config):
    """Fine-tune model and validate."""
    
    # Get historical data (last 48h from train_data)
    train_end = train_df['date'].max()
    historical_start = train_end - pd.Timedelta(hours=config['HISTORICAL_WINDOW_HOURS'])
    historical = train_df[train_df['date'] >= historical_start].copy()
    
    # Get current session data
    session_start = trade_df['date'].min()
    current_session = trade_df[
        (trade_df['date'] >= session_start) & 
        (trade_df['date'] <= current_time)
    ].copy()
    
    # Merge
    finetune_data = pd.concat([historical, current_session], ignore_index=True)
    finetune_data = finetune_data.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Recreate day index for the merged data
    unique_dates = sorted(finetune_data['date'].unique())
    date_to_day = {date: idx for idx, date in enumerate(unique_dates)}
    finetune_data['day'] = finetune_data['date'].map(date_to_day)
    
    # Split train/validation
    split_idx = int(len(unique_dates) * (1 - config['VALIDATION_SPLIT']))
    
    ft_train = finetune_data[finetune_data['date'].isin(unique_dates[:split_idx])].copy()
    ft_val = finetune_data[finetune_data['date'].isin(unique_dates[split_idx:])].copy()
    
    # CRITICAL: Reset day indices for each split to start from 0
    ft_train_dates = sorted(ft_train['date'].unique())
    ft_val_dates = sorted(ft_val['date'].unique())
    
    ft_train_date_to_day = {date: idx for idx, date in enumerate(ft_train_dates)}
    ft_val_date_to_day = {date: idx for idx, date in enumerate(ft_val_dates)}
    
    ft_train['day'] = ft_train['date'].map(ft_train_date_to_day)
    ft_val['day'] = ft_val['date'].map(ft_val_date_to_day)
    
    # Evaluate original
    original_score = evaluate_model(model, ft_val, config)
    
    # Clone model by saving and loading (deepcopy doesn't work with SB3 models)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name
        model.save(tmp_path)
        model_ft = PPO.load(tmp_path)
    os.remove(tmp_path)
    
    # Set fine-tuning learning rate
    model_ft.learning_rate = config['FINETUNE_LR']
    ft_env = create_env(ft_train, config)
    model_ft.set_env(ft_env)
    model_ft.learn(
        total_timesteps=config['FINETUNE_STEPS'], 
        reset_num_timesteps=False, 
        progress_bar=False
    )
    
    # Evaluate fine-tuned
    finetuned_score = evaluate_model(model_ft, ft_val, config)
    
    # Decision
    threshold = original_score * config['ROLLBACK_THRESHOLD']
    accepted = finetuned_score >= threshold
    
    result = {
        'original_score': original_score,
        'finetuned_score': finetuned_score,
        'accepted': accepted,
        'threshold': threshold,
        'train_rows': len(ft_train),
        'val_rows': len(ft_val),
    }
    
    return (model_ft if accepted else model), result


# ==================== MAIN SIMULATION ====================

def run_simulation():
    """Run complete simulation."""
    
    print("=" * 80)
    print("PRODUCTION FINE-TUNING SIMULATION")
    print("=" * 80)
    
    # Setup
    output_dir = Path(CONFIG['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df, trade_df = load_data()
    trading_times = get_trading_timestamps(trade_df, CONFIG['TRADE_INTERVAL_HOURS'])
    
    # Load model
    print(f"\n🤖 Loading model: {CONFIG['TRAINED_MODEL']}")
    model = PPO.load(CONFIG['TRAINED_MODEL'])
    print("✓ Model loaded\n")
    
    # Initialize
    cash = CONFIG['INITIAL_CASH']
    holdings = np.zeros(CONFIG['STOCK_DIM'])
    portfolio_value = CONFIG['INITIAL_CASH']
    
    portfolio_history = []
    finetune_results = []
    trade_logs = []
    
    # Simulation
    print(f"🚀 Starting simulation: {len(trading_times)} cycles\n")
    start_time = time.time()
    
    for idx, timestamp in enumerate(tqdm(trading_times, desc="Simulating")):
        cycle = idx + 1
        
        # Get market data
        data = trade_df[trade_df['date'] == timestamp].sort_values('tic')
        if len(data) == 0:
            continue
        
        prices = data['close'].values
        
        # Build state
        state = [cash] + list(prices) + list(holdings)
        for tech in TECH_INDICATORS:
            state.extend(data[tech].values)
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        
        # Predict
        predict_start = time.time()
        action, _ = model.predict(state, deterministic=True)
        predict_time = time.time() - predict_start
        
        actions = (action[0] * CONFIG['HMAX']).astype(int)
        
        # Get stock tickers for logging
        tickers = data['tic'].values
        
        # Log decisions for each stock
        stock_decisions = []
        for i, (ticker, act) in enumerate(zip(tickers, actions)):
            decision = 'HOLD'
            if act < 0:
                decision = 'SELL'
            elif act > 0:
                decision = 'BUY'
            
            stock_decisions.append({
                'ticker': ticker,
                'action': act,
                'decision': decision,
                'current_price': prices[i],
                'current_holdings': holdings[i],
            })
        
        # Execute trades
        trade_summary = []
        for i, act in enumerate(actions):
            shares_traded = 0
            trade_value = 0
            
            if act < 0:  # Sell
                sell_shares = min(abs(act), holdings[i])
                if sell_shares > 0:
                    trade_value = sell_shares * prices[i] * (1 - CONFIG['TRANSACTION_COST_PCT'])
                    cash += trade_value
                    holdings[i] -= sell_shares
                    shares_traded = -sell_shares
            elif act > 0:  # Buy
                cost = act * prices[i] * (1 + CONFIG['TRANSACTION_COST_PCT'])
                if cost <= cash:
                    cash -= cost
                    holdings[i] += act
                    shares_traded = act
                    trade_value = cost
            
            if shares_traded != 0:
                trade_summary.append({
                    'ticker': tickers[i],
                    'shares': shares_traded,
                    'value': trade_value,
                    'price': prices[i],
                })
        
        portfolio_value = cash + np.sum(holdings * prices)
        
        # Log trade details
        trade_log = {
            'timestamp': timestamp,
            'cycle': cycle,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'stocks_value': np.sum(holdings * prices),
            'predict_time_ms': predict_time * 1000,
            'num_trades': len(trade_summary),
            'decisions': stock_decisions,
            'trades': trade_summary,
            'holdings': holdings.copy(),
        }
        trade_logs.append(trade_log)
        
        portfolio_history.append({
            'timestamp': timestamp,
            'cycle': cycle,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'stocks_value': np.sum(holdings * prices),
        })
        
        # Fine-tune after each trade (except first)
        finetune_time = 0
        if idx > 0:
            finetune_start = time.time()
            model, ft_result = finetune_model(model, train_df, trade_df, timestamp, CONFIG)
            finetune_time = time.time() - finetune_start
            
            ft_result['timestamp'] = timestamp
            ft_result['cycle'] = cycle
            ft_result['finetune_time_sec'] = finetune_time
            finetune_results.append(ft_result)
            
            # Update trade log with finetune time
            trade_log['finetune_time_sec'] = finetune_time
    
    elapsed = time.time() - start_time
    
    # Results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"💼 Final value: ${portfolio_value:,.2f}")
    print(f"📈 Return: {(portfolio_value - CONFIG['INITIAL_CASH']) / CONFIG['INITIAL_CASH'] * 100:.2f}%")
    
    if finetune_results:
        accepted = sum(1 for r in finetune_results if r['accepted'])
        print(f"\n🔄 Fine-tuning:")
        print(f"   Total: {len(finetune_results)}")
        print(f"   Accepted: {accepted} ({accepted/len(finetune_results)*100:.1f}%)")
        print(f"   Rejected: {len(finetune_results) - accepted}")
    
    # Save
    pd.DataFrame(portfolio_history).to_csv(output_dir / 'portfolio_history.csv', index=False)
    pd.DataFrame(finetune_results).to_csv(output_dir / 'finetune_results.csv', index=False)
    
    print(f"\n✓ Saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_simulation()

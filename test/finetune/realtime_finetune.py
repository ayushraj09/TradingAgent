"""
Real-Time Fine-Tuning System

Continuously:
1. Downloads latest market data from Alpaca
2. Stores in MongoDB
3. Fine-tunes PPO model with last 48 hours of data
4. Validates and accepts/rejects fine-tuned model
5. Makes trading decisions with explainability (SHAP + LIME)

Usage:
    python realtime_finetune.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add FinRL to path
sys.path.append(str(Path(__file__).parent / 'FinRL'))

from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.config import INDICATORS
from stable_baselines3.common.vec_env import DummyVecEnv

# MongoDB
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: pymongo not installed")
    MONGODB_AVAILABLE = False

# Explainability
from explainable_drl.explainable_agent import ExplainableAgent
from explainable_drl.utils import get_feature_names

# ==================== CONFIGURATION ====================

CONFIG = {
    # Alpaca API
    'API_KEY': "PKPGA7BQIFZ7UV3V5ZYUWEXPUY",
    'API_SECRET': "HRvDc53DYAP2gJZbxn71MRCyZYnm5G5PFpCcAbhipf8Y",
    'API_BASE_URL': 'https://paper-api.alpaca.markets',
    
    # MongoDB
    'MONGO_URI': 'mongodb://localhost:27017/',
    'MONGO_DB': 'finrl_trading',
    'MONGO_COLLECTION': 'market_data_1min',
    
    # Model
    'TRAINED_MODEL': 'FinRL/examples/trained_models/agent_ppo.zip',
    'OUTPUT_DIR': 'realtime_finetune_results',
    
    # Trading
    'TICKERS': [
        'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
        'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD',
        'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'UNH', 'V', 'VZ', 'WMT'
    ],
    'STOCK_DIM': 30,
    'HMAX': 10000,
    'INITIAL_CASH': 1_000_000,
    
    # Fine-tuning
    'FINETUNE_INTERVAL_HOURS': 2,
    'FINETUNE_LR': 1e-5,
    'FINETUNE_STEPS': 2000,
    'LOOKBACK_HOURS': 48,
    'VALIDATION_SPLIT': 0.2,
    'ROLLBACK_THRESHOLD': 0.95,
    
    # Environment
    'TRANSACTION_COST_PCT': 0.001,
    'REWARD_SCALING': 1e-4,
    
    # Explainability
    'ENABLE_EXPLANATIONS': True,
    'SHAP_BACKGROUND_SAMPLES': 500,  # Reduced for faster real-time processing
    'EXPLAIN_TOP_N': 5,
    
    # Data Collection
    'INDICATORS': INDICATORS,
}

# ==================== MONGODB FUNCTIONS ====================

def connect_mongodb():
    """Connect to MongoDB and return collection."""
    if not MONGODB_AVAILABLE:
        raise ImportError("pymongo required. Install with: pip install pymongo")
    
    try:
        client = MongoClient(CONFIG['MONGO_URI'], serverSelectionTimeoutMS=5000)
        client.server_info()
        
        db = client[CONFIG['MONGO_DB']]
        collection = db[CONFIG['MONGO_COLLECTION']]
        
        print(f"✓ MongoDB connected: {CONFIG['MONGO_DB']}.{CONFIG['MONGO_COLLECTION']}")
        return collection
    except ConnectionFailure as e:
        print(f"✗ MongoDB connection failed: {e}")
        raise


def fetch_latest_data_alpaca_realtime():
    """Fetch real-time market data from Alpaca (like paper trading) and store in MongoDB."""
    
    print("\n" + "="*80)
    print("FETCHING REAL-TIME MARKET DATA")
    print("="*80)
    
    # Initialize Alpaca Processor
    alpaca = AlpacaProcessor(
        API_KEY=CONFIG['API_KEY'],
        API_SECRET=CONFIG['API_SECRET'],
        API_BASE_URL=CONFIG['API_BASE_URL']
    )
    
    try:
        # Fetch latest data using the same method as paper trading
        print(f"📥 Fetching latest 1Min data for {len(CONFIG['TICKERS'])} tickers...")
        
        price, tech, turbulence = alpaca.fetch_latest_data(
            ticker_list=CONFIG['TICKERS'],
            time_interval='1Min',
            tech_indicator_list=CONFIG['INDICATORS']
        )
        
        # Build DataFrame from fetched data
        current_time = datetime.utcnow()
        
        records = []
        for i, ticker in enumerate(CONFIG['TICKERS']):
            record = {
                'date': current_time,
                'tic': ticker,
                'close': price[i],
                'vixy': turbulence,  # VIX/turbulence indicator
            }
            
            # Add technical indicators
            tech_start_idx = i * len(CONFIG['INDICATORS'])
            for j, indicator in enumerate(CONFIG['INDICATORS']):
                record[indicator] = tech[tech_start_idx + j]
            
            records.append(record)
        
        df = pd.DataFrame(records)
        print(f"✓ Fetched real-time data: {len(df)} records at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Store in MongoDB
        collection = connect_mongodb()
        
        inserted = 0
        for record in records:
            try:
                collection.insert_one(record)
                inserted += 1
            except:
                pass  # Skip duplicates
        
        print(f"✓ Inserted: {inserted} new records to MongoDB")
        
        return df, price, tech, turbulence
        
    except Exception as e:
        print(f"✗ Real-time fetch error: {e}")
        print(f"   Falling back to historical data download...")
        return fetch_latest_data_alpaca_historical()


def fetch_latest_data_alpaca_historical():
    """Download historical market data from Alpaca (fallback method)."""
    
    # Initialize Alpaca
    alpaca = AlpacaProcessor(
        API_KEY=CONFIG['API_KEY'],
        API_SECRET=CONFIG['API_SECRET'],
        API_BASE_URL=CONFIG['API_BASE_URL']
    )
    
    # Download last 2 complete days (to avoid API restrictions)
    today = datetime.utcnow().date()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=1)
    
    alpaca.start = start_date.strftime('%Y-%m-%d')
    alpaca.end = end_date.strftime('%Y-%m-%d')
    alpaca.time_interval = '1Min'
    
    print(f"📥 Downloading historical: {start_date} to {end_date}")
    
    try:
        # Download raw data
        df_raw = alpaca.download_data(
            start_date=alpaca.start,
            end_date=alpaca.end,
            ticker_list=CONFIG['TICKERS'],
            time_interval='1Min'
        )
        
        if df_raw is None or len(df_raw) == 0:
            print("⚠️  No data from Alpaca")
            return None, None, None, None
        
        print(f"✓ Downloaded: {len(df_raw):,} records")
        
        # Process data
        if 'date' in df_raw.columns:
            df_raw.rename(columns={'date': 'timestamp'}, inplace=True)
        
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True, errors='coerce')
        
        # Clean and add indicators
        processed_df = alpaca.clean_data(df_raw)
        processed_df = alpaca.add_vix(processed_df)
        processed_df = alpaca.add_technical_indicator(processed_df, CONFIG['INDICATORS'])
        processed_df = processed_df.ffill().bfill()
        
        # Convert to timezone-naive for MongoDB
        if 'timestamp' in processed_df.columns:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
            if processed_df['timestamp'].dt.tz is not None:
                processed_df['timestamp'] = processed_df['timestamp'].dt.tz_localize(None)
            processed_df.rename(columns={'timestamp': 'date'}, inplace=True)
        
        print(f"✓ Processed: {len(processed_df):,} records")
        
        # Store in MongoDB
        collection = connect_mongodb()
        records = processed_df.to_dict('records')
        
        inserted = 0
        for record in records:
            try:
                collection.insert_one(record)
                inserted += 1
            except:
                pass  # Skip duplicates
        
        print(f"✓ Inserted: {inserted:,} new records to MongoDB")
        
        return processed_df, None, None, None
        
    except Exception as e:
        print(f"✗ Data fetch error: {e}")
        return None, None, None, None


def load_data_from_mongodb(hours=48):
    """Load last N hours of data from MongoDB."""
    collection = connect_mongodb()
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    cursor = collection.find(
        {'date': {'$gte': cutoff}},
        {'_id': 0}
    ).sort('date', ASCENDING)
    
    df = pd.DataFrame(list(cursor))
    
    if len(df) == 0:
        print(f"⚠️  No data in MongoDB for last {hours} hours")
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df):,} records from MongoDB ({hours}h window)")
    return df


# ==================== ENVIRONMENT FUNCTIONS ====================

def create_env(df, config):
    """Create StockTradingEnv from DataFrame."""
    
    # Create day index
    unique_dates = sorted(df['date'].unique())
    date_to_day = {date: idx for idx, date in enumerate(unique_dates)}
    df['day'] = df['date'].map(date_to_day)
    
    state_space = 1 + 2 * config['STOCK_DIM'] + len(CONFIG['INDICATORS']) * config['STOCK_DIM']
    
    df_indexed = df.copy().sort_values(['day', 'tic']).set_index('day')
    
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
        tech_indicator_list=CONFIG['INDICATORS'],
        print_verbosity=100000,
    )
    
    return DummyVecEnv([lambda: env])


def evaluate_model(model, df, config):
    """Evaluate model performance."""
    env = create_env(df, config)
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    
    return total_reward


# ==================== FINE-TUNING ====================

def finetune_model(model, lookback_hours=48):
    """Fine-tune model with latest data from MongoDB."""
    
    print("\n" + "="*80)
    print("FINE-TUNING MODEL")
    print("="*80)
    
    # Load data from MongoDB
    df = load_data_from_mongodb(hours=lookback_hours)
    
    if df is None or len(df) < 100:
        print("✗ Insufficient data for fine-tuning")
        return model, None
    
    # Split train/validation
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * (1 - CONFIG['VALIDATION_SPLIT']))
    
    train_df = df[df['date'].isin(unique_dates[:split_idx])].copy()
    val_df = df[df['date'].isin(unique_dates[split_idx:])].copy()
    
    # Reset day indices
    train_dates = sorted(train_df['date'].unique())
    val_dates = sorted(val_df['date'].unique())
    
    train_date_to_day = {date: idx for idx, date in enumerate(train_dates)}
    val_date_to_day = {date: idx for idx, date in enumerate(val_dates)}
    
    train_df['day'] = train_df['date'].map(train_date_to_day)
    val_df['day'] = val_df['date'].map(val_date_to_day)
    
    print(f"📊 Train: {len(train_df):,} records ({len(train_dates)} days)")
    print(f"📊 Val: {len(val_df):,} records ({len(val_dates)} days)")
    
    # Evaluate original
    print("🧪 Evaluating original model...")
    original_score = evaluate_model(model, val_df, CONFIG)
    print(f"   Original score: {original_score:.2f}")
    
    # Clone model
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name
        model.save(tmp_path)
        model_ft = PPO.load(tmp_path)
    os.remove(tmp_path)
    
    # Fine-tune
    print(f"🔄 Fine-tuning ({CONFIG['FINETUNE_STEPS']} steps, lr={CONFIG['FINETUNE_LR']})...")
    model_ft.learning_rate = CONFIG['FINETUNE_LR']
    ft_env = create_env(train_df, CONFIG)
    model_ft.set_env(ft_env)
    model_ft.learn(
        total_timesteps=CONFIG['FINETUNE_STEPS'],
        reset_num_timesteps=False,
        progress_bar=False
    )
    
    # Evaluate fine-tuned
    print("🧪 Evaluating fine-tuned model...")
    finetuned_score = evaluate_model(model_ft, val_df, CONFIG)
    print(f"   Fine-tuned score: {finetuned_score:.2f}")
    
    # Decision
    threshold = original_score * CONFIG['ROLLBACK_THRESHOLD']
    accepted = finetuned_score >= threshold
    
    result = {
        'timestamp': datetime.utcnow(),
        'original_score': original_score,
        'finetuned_score': finetuned_score,
        'threshold': threshold,
        'accepted': accepted,
        'improvement': ((finetuned_score - original_score) / abs(original_score) * 100) if original_score != 0 else 0
    }
    
    if accepted:
        print(f"✅ ACCEPTED (+{result['improvement']:.2f}%)")
        return model_ft, result
    else:
        print(f"❌ REJECTED ({result['improvement']:.2f}%)")
        return model, result


# ==================== TRADING DECISION ====================

def make_trading_decision(model, price, tech, turbulence, cash, stocks, stocks_cd):
    """Make a trading decision using the model with explainability."""
    
    # Build state vector (matching paper trading logic)
    turbulence_bool = 1 if turbulence >= 30 else 0  # turbulence threshold
    turbulence_scaled = (sigmoid_sign(turbulence, 30) * 2 ** -5).astype(np.float32)
    tech_scaled = tech * 2 ** -7
    
    amount = np.array(cash * (2 ** -12), dtype=np.float32)
    scale = np.array(2 ** -6, dtype=np.float32)
    
    state = np.hstack((
        amount,
        turbulence_scaled,
        turbulence_bool,
        price * scale,
        stocks * scale,
        stocks_cd,
        tech_scaled,
    )).astype(np.float32)
    
    # Handle NaN/Inf
    state[np.isnan(state)] = 0.0
    state[np.isinf(state)] = 0.0
    
    # Get prediction with explanations
    if CONFIG['ENABLE_EXPLANATIONS'] and hasattr(model, 'predict_with_explanation'):
        result = model.predict_with_explanation(state.reshape(1, -1), explain_method='all')
        action = result['action']  # Already scaled by ExplainableAgent
        explanations = result['methods']
        
        # Display explanations
        if explanations:
            print("\n📊 DECISION EXPLANATIONS:")
            
            # SHAP top features
            if 'shap' in explanations:
                shap_data = explanations['shap']
                top_contribs = shap_data['top_contributors']
                print("   🔍 SHAP Top 5 Features:")
                for i in range(min(5, len(top_contribs))):
                    feature_name, value = top_contribs[i]
                    print(f"      {i+1}. {feature_name}: {value:.4f}")
            
            # LIME top features
            if 'lime' in explanations:
                lime_data = explanations['lime']
                feature_contribs = lime_data.get('feature_contributions', [])
                print("   🍋 LIME Top 5 Features:")
                for i in range(min(5, len(feature_contribs))):
                    feature_desc, weight = feature_contribs[i]
                    print(f"      {i+1}. {feature_desc}: {weight:.4f}")
    else:
        # Vanilla prediction
        action, _ = model.predict(state.reshape(1, -1), deterministic=True)
        action = action.flatten()
        # Scale actions
        action = (action * CONFIG['HMAX']).astype(int)
        explanations = None
    
    # Display trading signals
    min_action = 10
    buy_signals = [(CONFIG['TICKERS'][i], action[i]) for i in range(len(action)) if action[i] > min_action]
    sell_signals = [(CONFIG['TICKERS'][i], action[i]) for i in range(len(action)) if action[i] < -min_action]
    
    print(f"\n📈 Trading Signals:")
    print(f"   BUY ({len(buy_signals)}): {buy_signals[:5]}")  # Show top 5
    print(f"   SELL ({len(sell_signals)}): {sell_signals[:5]}")
    print(f"   Turbulence: {turbulence:.2f} {'⚠️ HIGH' if turbulence_bool else '✓ Normal'}")
    
    return action, explanations


def sigmoid_sign(ary, thresh):
    """Sigmoid transformation for turbulence."""
    def sigmoid(x):
        return 1 / (1 + np.exp(-x * np.e)) - 0.5
    return sigmoid(ary / thresh) * thresh


# ==================== MAIN LOOP ====================

def realtime_finetune_loop():
    """Main real-time fine-tuning loop."""
    
    print("=" * 80)
    print("REAL-TIME FINE-TUNING SYSTEM")
    print("=" * 80)
    
    # Setup
    output_dir = Path(CONFIG['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load initial model
    print(f"\n🤖 Loading model: {CONFIG['TRAINED_MODEL']}")
    
    # Load PPO model first
    ppo_model = PPO.load(CONFIG['TRAINED_MODEL'])
    print("✓ PPO model loaded")
    
    if CONFIG['ENABLE_EXPLANATIONS']:
        # Wrap with ExplainableAgent
        model = ExplainableAgent(ppo_model, stock_dim=CONFIG['STOCK_DIM'], hmax=CONFIG['HMAX'])
        
        # Train explainers on historical data
        print(f"\n📊 Training SHAP + LIME explainers ({CONFIG['SHAP_BACKGROUND_SAMPLES']} samples)...")
        model.train_explainers(
            train_data_path='FinRL/examples/train_data.csv',
            n_samples=CONFIG['SHAP_BACKGROUND_SAMPLES']
        )
        print("✓ ExplainableAgent initialized (SHAP + LIME)")
    else:
        model = ppo_model
        print("✓ Using PPO model without explanations")
    
    # Initialize tracking
    finetune_history = []
    trading_history = []
    last_finetune = datetime.utcnow() - timedelta(hours=CONFIG['FINETUNE_INTERVAL_HOURS'])
    cycle = 0
    
    # Initialize portfolio state
    cash = CONFIG['INITIAL_CASH']
    stocks = np.zeros(CONFIG['STOCK_DIM'])
    stocks_cd = np.zeros(CONFIG['STOCK_DIM'])
    portfolio_value = CONFIG['INITIAL_CASH']
    
    print(f"\n🚀 Starting real-time loop (fine-tune every {CONFIG['FINETUNE_INTERVAL_HOURS']}h)")
    print("="*80)
    
    try:
        while True:
            cycle += 1
            current_time = datetime.utcnow()
            
            print(f"\n{'='*80}")
            print(f"CYCLE {cycle} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # 1. Fetch latest data (try real-time first, fallback to historical)
            df, price, tech, turbulence = fetch_latest_data_alpaca_realtime()
            
            # 2. Make trading decision if we have real-time data
            if price is not None and tech is not None and turbulence is not None:
                print("\n💡 Making trading decision with current model...")
                
                action, explanations = make_trading_decision(
                    model, price, tech, turbulence, 
                    cash, stocks, stocks_cd
                )
                
                # Update portfolio value
                portfolio_value = cash + np.sum(stocks * price)
                
                # Log trading decision
                trading_log = {
                    'timestamp': current_time,
                    'cycle': cycle,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'turbulence': turbulence,
                    'action': action.tolist(),
                }
                
                if explanations:
                    trading_log['explanations'] = explanations
                
                trading_history.append(trading_log)
                
                # Save trading history
                pd.DataFrame(trading_history).to_csv(
                    output_dir / 'trading_history.csv',
                    index=False
                )
                
                print(f"💼 Portfolio: ${portfolio_value:,.2f} (Cash: ${cash:,.2f}, Stocks: ${np.sum(stocks * price):,.2f})")
            
            # 3. Check if it's time to fine-tune
            time_since_finetune = (current_time - last_finetune).total_seconds() / 3600
            
            if time_since_finetune >= CONFIG['FINETUNE_INTERVAL_HOURS']:
                print(f"\n⏰ Time to fine-tune (last: {time_since_finetune:.1f}h ago)")
                
                # Fine-tune with explainability check
                if CONFIG['ENABLE_EXPLANATIONS'] and not hasattr(model, 'shap_explainer'):
                    print("📊 Training explainers...")
                    # Load data for explainer training
                    df = load_data_from_mongodb(hours=CONFIG['LOOKBACK_HOURS'])
                    if df is not None and len(df) > CONFIG['SHAP_BACKGROUND_SAMPLES']:
                        # Sample background data
                        sample_df = df.sample(n=CONFIG['SHAP_BACKGROUND_SAMPLES'], random_state=42)
                        # Build state vectors (simplified - you may need to adjust)
                        # This is a placeholder - actual implementation depends on your state building
                        model.train_explainers_from_df(sample_df)
                        print("✓ Explainers trained")
                
                model, ft_result = finetune_model(model, lookback_hours=CONFIG['LOOKBACK_HOURS'])
                
                if ft_result:
                    finetune_history.append(ft_result)
                    last_finetune = current_time
                    
                    # Save results
                    pd.DataFrame(finetune_history).to_csv(
                        output_dir / 'finetune_history.csv',
                        index=False
                    )
                    
                    # Save model if accepted
                    if ft_result['accepted']:
                        model_path = output_dir / f'model_cycle_{cycle}.zip'
                        model.save(str(model_path))
                        print(f"💾 Model saved: {model_path}")
            
            # 3. Wait before next cycle
            wait_time = 300  # 5 minutes
            print(f"\n⏸️  Waiting {wait_time}s before next cycle...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total cycles: {cycle}")
    print(f"Trading decisions: {len(trading_history)}")
    print(f"Fine-tuning sessions: {len(finetune_history)}")
    
    if trading_history:
        final_value = trading_history[-1]['portfolio_value']
        total_return = (final_value - CONFIG['INITIAL_CASH']) / CONFIG['INITIAL_CASH'] * 100
        print(f"\n💰 Portfolio Performance:")
        print(f"   Initial: ${CONFIG['INITIAL_CASH']:,.2f}")
        print(f"   Final: ${final_value:,.2f}")
        print(f"   Return: {total_return:+.2f}%")
    
    if finetune_history:
        accepted = sum(1 for r in finetune_history if r['accepted'])
        print(f"\n🔄 Fine-tuning:")
        print(f"   Accepted: {accepted}/{len(finetune_history)} ({accepted/len(finetune_history)*100:.1f}%)")
        avg_improvement = np.mean([r['improvement'] for r in finetune_history])
        print(f"   Avg improvement: {avg_improvement:.2f}%")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"   - trading_history.csv (all trading decisions)")
    print(f"   - finetune_history.csv (fine-tuning performance)")


if __name__ == "__main__":
    realtime_finetune_loop()

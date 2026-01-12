"""
Configuration module for paper trading system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from finrl.config import INDICATORS


def load_config(enable_explanations=True):
    """
    Load configuration from .env file and return config dictionary.
    
    Args:
        enable_explanations: Whether to enable SHAP/LIME explanations
    
    Returns:
        dict: Configuration dictionary
    """
    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    
    # Load API credentials
    API_KEY = os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('ALPACA_API_SECRET')
    API_BASE_URL = os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    if not API_KEY or not API_SECRET:
        raise ValueError("Missing ALPACA_API_KEY or ALPACA_API_SECRET in .env file")
    
    # DOW 30 tickers (excluding VIXY)
    TICKERS = [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
    ]
    
    TECH_INDICATORS = INDICATORS
    STOCK_DIM = len(TICKERS)
    
    # Calculate state dimensions (matches StockTradingEnv)
    # State: 1 (cash) + 30 (prices) + 30 (stocks) + 240 (tech indicators) = 301
    state_dim = 1 + 2 * STOCK_DIM + len(TECH_INDICATORS) * STOCK_DIM
    action_dim = STOCK_DIM
    
    CONFIG = {
        # Alpaca API
        'API_KEY': API_KEY,
        'API_SECRET': API_SECRET,
        'API_BASE_URL': API_BASE_URL,
        
        # Tickers
        'TICKERS': TICKERS,
        'STOCK_DIM': STOCK_DIM,
        'TECH_INDICATORS': TECH_INDICATORS,
        
        # Model paths
        'TRAINED_MODEL': 'trained_models/agent_ppo.zip',
        'OUTPUT_DIR': 'production_paper_trading_results',
        
        # Trading parameters
        'INITIAL_CASH': 1_000_000,
        'HMAX': 100,
        'MAX_STOCK': 100,
        'TRANSACTION_COST_PCT': 0.001,
        'REWARD_SCALING': 1e-4,
        'TURBULENCE_THRESHOLD': 500,
        'MIN_ACTION_THRESHOLD': 10,
        
        # Trading timing
        'TIME_INTERVAL_MIN': 1,  # Trade every minute
    'INITIAL_TRADE_DELAY_MIN': 1,  # Wait 1 min after market open (instant for testing)
        'FINETUNE_INTERVAL_HOURS': 2,
        'FINETUNE_LOOKBACK_HOURS': 48,
        'FINETUNE_LR': 1e-5,
        'FINETUNE_STEPS': 2000,
        'VALIDATION_SPLIT': 0.2,
        'ROLLBACK_THRESHOLD': 0.95,
        
        # Data storage
        'DATA_CSV': 'production_paper_trading_data.csv',
        
        # Explainability
        'ENABLE_EXPLANATIONS': enable_explanations,
        'SHAP_BACKGROUND_SAMPLES': 500,
    
    # State/action dimensions
    'state_dim': state_dim,
    'action_dim': action_dim,
    }
    
    return CONFIG


def print_config(config):
    """Print configuration summary."""
    print("\n📋 CONFIGURATION")
    print(f"Explainability: {config['ENABLE_EXPLANATIONS']}")
    print("="*80)

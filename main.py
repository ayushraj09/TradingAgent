#!/usr/bin/env python3
"""
Production Paper Trading System - Main Entry Point

Usage:
    python main.py                    # Run with explainability enabled (default)
    python main.py --no-explain       # Run without explainability
    python main.py --help             # Show help
"""
import sys
import argparse
import warnings
from pathlib import Path

# Add FinRL to path
PROJECT_ROOT = Path(__file__).parent.parent
finrl_path = PROJECT_ROOT / 'FinRL'
if str(finrl_path) not in sys.path:
    sys.path.append(str(finrl_path))

warnings.filterwarnings('ignore')

from config import load_config, print_config
from paper_trader import ProductionPaperTrading


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Production Paper Trading System with Optional Explainability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with explainability enabled (SHAP + LIME)
  python main.py
  
  # Run without explainability (faster, less overhead)
  python main.py --no-explain
  
  # Use custom model path
  python main.py --model path/to/model.zip
  
  # Set custom output directory
  python main.py --output my_results/
        """
    )
    
    parser.add_argument(
        '--no-explain',
        action='store_true',
        help='Disable SHAP/LIME explainability (faster, less memory)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='trained_models/agent_ppo.zip',
        help='Path to trained PPO model (default: trained_models/agent_ppo.zip)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='production_paper_trading_results',
        help='Output directory for results (default: production_paper_trading_results)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Trading interval in minutes (default: 1)'
    )
    
    parser.add_argument(
        '--finetune-interval',
        type=int,
        default=2,
        help='Fine-tuning interval in hours (default: 2)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("\n" + "="*80)
    print("PRODUCTION PAPER TRADING SYSTEM")
    print("="*80)
    print(f"Explainability: {'DISABLED' if args.no_explain else 'ENABLED (SHAP + LIME)'}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Trading interval: {args.interval} minute(s)")
    print(f"Fine-tuning interval: {args.finetune_interval} hour(s)")
    print("="*80)
    
    # Load configuration
    enable_explanations = not args.no_explain
    config = load_config(enable_explanations=enable_explanations)
    
    # Override config with CLI arguments
    config['TRAINED_MODEL'] = args.model
    config['OUTPUT_DIR'] = args.output
    config['TIME_INTERVAL_MIN'] = args.interval
    config['FINETUNE_INTERVAL_HOURS'] = args.finetune_interval
    
    # Print configuration
    print_config(config)
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print(f"   Please train a model first or specify correct path with --model")
        sys.exit(1)
    
    # Check .env file exists
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        print(f"\n❌ ERROR: .env file not found at {env_path}")
        print(f"   Please create .env with Alpaca API credentials")
        sys.exit(1)
    
    # Initialize and run paper trader
    try:
        trader = ProductionPaperTrading(
            config=config,
            model_path=str(model_path)
        )
        
        print("\n✅ Initialization complete")
        print("🚀 Starting paper trading loop...")
        print("   Press Ctrl+C to stop\n")
        
        trader.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Trading stopped by user")
        print("✓ Session ended gracefully")
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

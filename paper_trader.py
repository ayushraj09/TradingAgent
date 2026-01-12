"""
Production Paper Trading class with explainability and fine-tuning.
"""
import sys
import time
import numpy as np
import pandas as pd
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor

from data_manager import (
    init_data_csv,
    fetch_historical_data_to_csv,
    append_latest_data_to_csv,
    load_recent_data_from_csv
)
from trading_utils import (
    get_state_from_alpaca,
    submit_order,
    finetune_model_with_validation
)


class ProductionPaperTrading:
    """
    Production paper trading with explainability and fine-tuning.
    """
    
    def __init__(self, config, model_path):
        self.config = config
        
        # Initialize Alpaca
        self.alpaca_processor = AlpacaProcessor(
            API_KEY=config['API_KEY'],
            API_SECRET=config['API_SECRET'],
            API_BASE_URL=config['API_BASE_URL']
        )
        
        self.alpaca = tradeapi.REST(
            config['API_KEY'],
            config['API_SECRET'],
            config['API_BASE_URL'],
            'v2'
        )
        
        # Load PPO model
        print(f"🤖 Loading PPO model: {model_path}")
        ppo_model = PPO.load(model_path)
        print("✓ PPO model loaded")
        
        # Wrap with ExplainableAgent if enabled
        if config['ENABLE_EXPLANATIONS']:
            # Add parent directory to path for explainable_drl import
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from explainable_drl.explainable_agent import ExplainableAgent
            
            self.model = ExplainableAgent(
                ppo_model, 
                stock_dim=config['STOCK_DIM'],
                hmax=config['MAX_STOCK']
            )
            print("✓ Wrapped with ExplainableAgent")
        else:
            self.model = ppo_model
            print("✓ Using PPO without explanations")
        
        # Initialize state
        self.tickers = config['TICKERS']
        self.stocks_cd = np.zeros(config['STOCK_DIM'])
        
        # Fine-tuning tracking
        self.last_finetune = datetime.utcnow() - timedelta(hours=config['FINETUNE_INTERVAL_HOURS'])
        self.finetune_history = []
        self.trading_history = []
        self.cycle = 0
        self.model_version = 'original'
        self.finetune_count = 0
        
        # Initialize data CSV
        print(f"\n📊 Initializing data collection...")
        init_data_csv(config['DATA_CSV'])
        
        # Fetch historical data
        fetch_historical_data_to_csv(
            self.alpaca,
            config['DATA_CSV'],
            required_days=2
        )
        
        # Train explainers if enabled
        if config['ENABLE_EXPLANATIONS']:
            self._train_explainers()
        
        print("✓ ProductionPaperTrading initialized")
    
    def _train_explainers(self):
        """Train SHAP and LIME explainers on historical data."""
        print(f"\n🔍 Training SHAP + LIME explainers...")
        
        try:
            # Load historical data for explainer training
            df_hist = load_recent_data_from_csv(
                self.config['DATA_CSV'],
                self.config,
                hours=self.config['FINETUNE_LOOKBACK_HOURS']
            )
            
            if df_hist is None or len(df_hist) < 100:
                print("⚠️  Insufficient data for explainer training")
                print("   Will train explainers after collecting more data")
                return
            
            # Save to temp file for training
            temp_path = Path(self.config['OUTPUT_DIR']) / 'temp_explainer_data.csv'
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df_hist.to_csv(temp_path, index=False)
            
            # Train explainers
            self.model.train_explainers(
                train_data_path=str(temp_path),
                n_samples=self.config['SHAP_BACKGROUND_SAMPLES']
            )
            
            # Clean up
            temp_path.unlink()
            
            print("✓ SHAP + LIME explainers trained")
            
        except Exception as e:
            print(f"⚠️  Failed to train explainers: {e}")
            print("   Will continue without explainability")
    
    def execute_trade(self):
        """
        Execute trading decision with optional explanations.
        """
        # Get current state (301 features)
        state, price, stocks, cash, turbulence, turbulence_bool, tech = get_state_from_alpaca(
            self.alpaca_processor, self.config
        )
        
        # Get prediction
        if self.config['ENABLE_EXPLANATIONS'] and hasattr(self.model, 'predict_with_explanation'):
            # Check if explainers are trained
            if self.model.shap_explainer is not None and self.model.lime_explainer is not None:
                result = self.model.predict_with_explanation(
                    state.reshape(1, -1),
                    explain_method='all'
                )
                action = result['action']  # Already scaled
                explanations = result['methods']
            else:
                # Fallback to regular predict
                action, _ = self.model.predict(state.reshape(1, -1), deterministic=True)
                action = action[0]
                action = (action * self.config['MAX_STOCK']).astype(int)
                explanations = None
        else:
            action, _ = self.model.predict(state.reshape(1, -1), deterministic=True)
            action = action[0]
            action = (action * self.config['MAX_STOCK']).astype(int)
            explanations = None
        
        # Update cooldown
        self.stocks_cd += 1
        
        # Execute trades
        decisions = []
        
        if turbulence_bool == 0:
            # Normal trading
            min_action = self.config['MIN_ACTION_THRESHOLD']
            threads = []
            
            # SELL orders
            sell_indices = np.where(action < -min_action)[0]
            for index in sell_indices:
                sell_num_shares = min(stocks[index], -action[index])
                qty = abs(int(sell_num_shares))
                respSO = []
                
                t = threading.Thread(
                    target=lambda q=qty, s=self.tickers[index]: submit_order(
                        self.alpaca, q, s, 'sell', respSO
                    )
                )
                t.start()
                threads.append(t)
                self.stocks_cd[index] = 0
                
                if qty > 0:
                    decisions.append({
                        'ticker': self.tickers[index],
                        'action': 'SELL',
                        'qty': qty,
                        'price': price[index],
                    })
            
            # Wait for sells
            for t in threads:
                t.join()
            
            # Update cash
            cash = float(self.alpaca.get_account().cash)
            
            # BUY orders
            threads = []
            buy_indices = np.where(action > min_action)[0]
            for index in buy_indices:
                tmp_cash = max(0, cash)
                buy_num_shares = min(tmp_cash // price[index], abs(int(action[index])))
                qty = abs(int(buy_num_shares)) if not np.isnan(buy_num_shares) else 0
                respSO = []
                
                t = threading.Thread(
                    target=lambda q=qty, s=self.tickers[index]: submit_order(
                        self.alpaca, q, s, 'buy', respSO
                    )
                )
                t.start()
                threads.append(t)
                self.stocks_cd[index] = 0
                
                if qty > 0:
                    decisions.append({
                        'ticker': self.tickers[index],
                        'action': 'BUY',
                        'qty': qty,
                        'price': price[index],
                    })
            
            # Wait for buys
            for t in threads:
                t.join()
            
            # HOLD
            hold_indices = np.where((action >= -min_action) & (action <= min_action))[0]
            for index in hold_indices:
                decisions.append({
                    'ticker': self.tickers[index],
                    'action': 'HOLD',
                    'qty': 0,
                    'price': price[index],
                })
        
        else:
            # High turbulence - liquidate all
            print("  ⚠️  HIGH TURBULENCE - Liquidating all positions")
            threads = []
            positions = self.alpaca.list_positions()
            
            for position in positions:
                side = 'sell' if position.side == 'long' else 'buy'
                qty = abs(int(float(position.qty)))
                respSO = []
                
                t = threading.Thread(
                    target=lambda q=qty, sym=position.symbol, s=side: submit_order(
                        self.alpaca, q, sym, s, respSO
                    )
                )
                t.start()
                threads.append(t)
                
                decisions.append({
                    'ticker': position.symbol,
                    'action': 'SELL_TURBULENCE',
                    'qty': qty,
                    'price': 0,
                })
            
            for t in threads:
                t.join()
            
            self.stocks_cd[:] = 0
        
        # Get final values
        cash = float(self.alpaca.get_account().cash)
        portfolio_value = float(self.alpaca.get_account().last_equity)
        
        # Append new data to CSV
        append_latest_data_to_csv(self.alpaca_processor, self.config['DATA_CSV'])
        
        return {
            'decisions': decisions,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'turbulence': turbulence,
            'turbulence_bool': turbulence_bool,
            'explanations': explanations,
        }
    
    def check_and_finetune(self):
        """Check if it's time to fine-tune and execute if needed."""
        current_time = datetime.utcnow()
        time_since_finetune = (current_time - self.last_finetune).total_seconds() / 3600
        
        if time_since_finetune >= self.config['FINETUNE_INTERVAL_HOURS']:
            print(f"\n⏰ Time to fine-tune (last: {time_since_finetune:.1f}h ago)")
            
            self.model, ft_result = finetune_model_with_validation(
                self.model,
                self.config['DATA_CSV'],
                self.config
            )
            
            if ft_result:
                self.finetune_history.append(ft_result)
                self.last_finetune = current_time
                
                # Save results
                output_dir = Path(self.config['OUTPUT_DIR'])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                pd.DataFrame(self.finetune_history).to_csv(
                    output_dir / 'finetune_history.csv',
                    index=False
                )
                
                if ft_result['accepted']:
                    self.finetune_count += 1
                    self.model_version = f'finetuned_v{self.finetune_count}'
                    
                    # Save fine-tuned model
                    model_path = output_dir / f'model_cycle_{self.cycle}.zip'
                    self.model.save(str(model_path))
                    print(f"💾 Saved fine-tuned model: {model_path}")
                    
                    # Retrain explainers on updated model
                    if self.config['ENABLE_EXPLANATIONS']:
                        print("  Re-training explainers on fine-tuned model...")
                        self._train_explainers()
    
    def square_off_all_positions(self):
        """Liquidate all positions before market close."""
        print("\n🔚 Squaring off all positions...")
        positions = self.alpaca.list_positions()
        
        if len(positions) == 0:
            print("   No positions to square off")
            return
        
        threads = []
        for position in positions:
            side = 'sell' if position.side == 'long' else 'buy'
            qty = abs(int(float(position.qty)))
            respSO = []
            
            t = threading.Thread(
                target=lambda q=qty, sym=position.symbol, s=side: submit_order(
                    self.alpaca, q, sym, s, respSO
                )
            )
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        print("✓ All positions squared off")
    
    def run(self):
        """Main trading loop."""
        # Wait for market to open
        clock = self.alpaca.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open.replace(tzinfo=timezone.utc) - 
                          clock.timestamp.replace(tzinfo=timezone.utc)).total_seconds()
            print(f"⏰ Market closed - waiting {int(time_to_open/60)} minutes...")
            time.sleep(time_to_open)
        
        # Wait initial delay after market open
        print(f"✅ Market opened - waiting {self.config['INITIAL_TRADE_DELAY_MIN']} minutes...")
        time.sleep(self.config['INITIAL_TRADE_DELAY_MIN'] * 60)
        
        print(f"\n🚀 Starting paper trading (Model: {self.model_version})")
        print("="*80)
        
        output_dir = Path(self.config['OUTPUT_DIR'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            while True:
                self.cycle += 1
                
                # Check market status
                clock = self.alpaca.get_clock()
                closing_time = clock.next_close.replace(tzinfo=timezone.utc).timestamp()
                curr_time = clock.timestamp.replace(tzinfo=timezone.utc).timestamp()
                time_to_close = closing_time - curr_time
                
                # Square off 15 min before close
                if time_to_close < (15 * 60):
                    self.square_off_all_positions()
                    print("🔚 Market closing soon - stopping trading")
                    break
                
                print(f"\n{'='*80}")
                print(f"CYCLE {self.cycle} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Model: {self.model_version} | Time to close: {int(time_to_close/60)} mins")
                print(f"{'='*80}")
                
                # Execute trade
                trade_result = self.execute_trade()
                
                # Log trade
                trade_log = {
                    'timestamp': datetime.utcnow(),
                    'cycle': self.cycle,
                    'portfolio_value': trade_result['portfolio_value'],
                    'cash': trade_result['cash'],
                    'turbulence': trade_result['turbulence'],
                    'num_trades': len([d for d in trade_result['decisions'] if d['action'] != 'HOLD']),
                }
                self.trading_history.append(trade_log)
                
                # Check and fine-tune
                self.check_and_finetune()
                
                # Save trading history
                pd.DataFrame(self.trading_history).to_csv(
                    output_dir / 'trading_history.csv',
                    index=False
                )
                
                # Wait for next interval
                time.sleep(self.config['TIME_INTERVAL_MIN'] * 60)
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        # Final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print final trading summary."""
        print(f"\n{'='*80}")
        print("TRADING SESSION SUMMARY")
        print(f"{'='*80}")
        print(f"Total cycles: {self.cycle}")
        print(f"Trading decisions: {len(self.trading_history)}")
        print(f"Fine-tuning sessions: {len(self.finetune_history)}")
        
        if self.trading_history:
            final_value = self.trading_history[-1]['portfolio_value']
            initial_value = self.trading_history[0]['portfolio_value']
            total_return = (final_value - initial_value) / initial_value * 100
            
            print(f"\n💰 Portfolio Performance:")
            print(f"   Initial: ${initial_value:,.2f}")
            print(f"   Final: ${final_value:,.2f}")
            print(f"   Return: {total_return:+.2f}%")
        
        if self.finetune_history:
            accepted = sum(1 for r in self.finetune_history if r['accepted'])
            avg_improvement = np.mean([r['improvement_pct'] for r in self.finetune_history])
            
            print(f"\n🔄 Fine-tuning:")
            print(f"   Accepted: {accepted}/{len(self.finetune_history)}")
            print(f"   Avg improvement: {avg_improvement:.2f}%")
        
        print(f"\n✓ Results saved to: {self.config['OUTPUT_DIR']}")
        print("="*80)

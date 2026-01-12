"""
Data management module for paper trading system.
Handles CSV-based data storage with historical backfill and real-time updates.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor


def init_data_csv(csv_path, tech_indicators=None):
    """Initialize CSV file for data collection."""
    # Use INDICATORS from config if not provided
    if tech_indicators is None:
        from finrl.config import INDICATORS
        tech_indicators = INDICATORS
        
    if not Path(csv_path).exists():
        # Create with all required columns
        columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'] + tech_indicators
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"✓ Initialized CSV: {csv_path}")
    else:
        existing_df = pd.read_csv(csv_path)
        print(f"✓ CSV exists: {csv_path} ({len(existing_df):,} records)")


def fetch_historical_data_to_csv(alpaca_api, csv_path, required_days=2):
    """
    Fetch historical 1-min data from Alpaca and populate CSV.
    Uses last 2 completed trading days to avoid API restrictions.
    """
    from finrl.config import INDICATORS
    
    print(f"\n📥 Fetching historical data for {required_days} trading days...")
    
    # Check if CSV already has sufficient data
    if Path(csv_path).exists():
        existing_df = pd.read_csv(csv_path)
        if len(existing_df) > 0:
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            span_hours = (existing_df['date'].max() - existing_df['date'].min()).total_seconds() / 3600
            age_hours = (datetime.utcnow() - existing_df['date'].max()).total_seconds() / 3600
            
            if span_hours >= 12 and age_hours < 24:  # At least 12h of recent data
                print(f"✓ CSV has {span_hours:.1f}h of data ({age_hours:.1f}h old)")
                print(f"  Skipping historical fetch")
                return
    
    try:
        # Calculate date range
        today = datetime.utcnow().date()
        end_date = today - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=2)  # 2 days before
        
        print(f"  Download range: {start_date} to {end_date}")
        
        # Get config for AlpacaProcessor
        from config import load_config
        config = load_config(enable_explanations=False)
        
        # Initialize Alpaca processor
        alpaca_processor = AlpacaProcessor(
            API_KEY=config['API_KEY'],
            API_SECRET=config['API_SECRET'],
            API_BASE_URL=config['API_BASE_URL']
        )
        
        # Download raw data
        print(f"  Downloading...")
        df_raw = alpaca_processor.download_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            ticker_list=config['TICKERS'],
            time_interval='1Min'
        )
        
        if df_raw is None or len(df_raw) == 0:
            print("  ⚠️  No data returned from Alpaca")
            return
        
        print(f"  ✓ Downloaded: {len(df_raw):,} records")
        
        # Process data
        if 'date' in df_raw.columns:
            df_raw.rename(columns={'date': 'timestamp'}, inplace=True)
        
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True, errors='coerce')
        
        # Set processor attributes for clean_data
        alpaca_processor.start = start_date.strftime('%Y-%m-%d')
        alpaca_processor.end = end_date.strftime('%Y-%m-%d')
        alpaca_processor.time_interval = '1Min'
        
        # Clean and add indicators
        df_clean = alpaca_processor.clean_data(df_raw)
        df_clean = df_clean.sort_values(by=['timestamp', 'tic']).reset_index(drop=True)
        df_clean = alpaca_processor.add_technical_indicator(df_clean, INDICATORS)
        df_clean = df_clean.ffill().bfill()
        
        print(f"  ✓ Processed: {len(df_clean):,} records")
        
        # Convert to timezone-naive
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        if df_clean['timestamp'].dt.tz is not None:
            df_clean['timestamp'] = df_clean['timestamp'].dt.tz_localize(None)
        
        df_clean.rename(columns={'timestamp': 'date'}, inplace=True)
        
        # Save to CSV
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'] + INDICATORS
        df_clean = df_clean[required_cols]
        df_clean.to_csv(csv_path, index=False)
        
        print(f"✅ Saved {len(df_clean):,} records to {csv_path}")
        print(f"   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
        
    except Exception as e:
        print(f"✗ Failed to fetch historical data: {e}")
        import traceback
        traceback.print_exc()


def append_latest_data_to_csv(alpaca_processor, csv_path):
    """
    Fetch latest 1-min data from Alpaca and append to CSV.
    Returns: DataFrame of new data (or None if failed/duplicate)
    """
    from finrl.config import INDICATORS
    from config import load_config
    
    config = load_config(enable_explanations=False)
    
    try:
        # Fetch latest data with technical indicators
        price, tech, turbulence = alpaca_processor.fetch_latest_data(
            ticker_list=config['TICKERS'],
            time_interval='1Min',
            tech_indicator_list=INDICATORS
        )
        
        if price is None:
            print("⚠️  No data fetched")
            return None
        
        # Get current timestamp (rounded to minute)
        current_time = datetime.utcnow().replace(second=0, microsecond=0)
        
        # Build DataFrame
        records = []
        for i, ticker in enumerate(config['TICKERS']):
            record = {
                'date': current_time,
                'tic': ticker,
                'open': price[i],  # Using close as proxy
                'high': price[i],
                'low': price[i],
                'close': price[i],
                'volume': 0,
            }
            
            # Add tech indicators
            for j, tech_name in enumerate(INDICATORS):
                idx = i * len(INDICATORS) + j
                record[tech_name] = tech[idx] if idx < len(tech) else 0
            
            records.append(record)
        
        df_new = pd.DataFrame(records)
        
        # Check for duplicates
        if Path(csv_path).exists():
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > 0:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                last_timestamp = existing_df['date'].max()
                
                if current_time <= last_timestamp:
                    print(f"  ⚠️  Data already exists for {current_time}")
                    return None
        
        # Append to CSV
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"  💾 Appended {len(df_new)} records at {current_time}")
        
        return df_new
        
    except Exception as e:
        print(f"✗ Error appending data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_recent_data_from_csv(csv_path, config, hours=48):
    """Load last N hours of data from CSV."""
    if not Path(csv_path).exists():
        print(f"⚠️  CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("⚠️  CSV is empty")
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Get last N hours
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df_filtered = df[df['date'] >= cutoff].copy()
    
    # Filter to only expected tickers
    df_filtered = df_filtered[df_filtered['tic'].isin(config['TICKERS'])]
    
    # Filter to complete timestamps (all 30 stocks)
    df_filtered = df_filtered.sort_values(['date', 'tic']).reset_index(drop=True)
    timestamp_counts = df_filtered.groupby('date')['tic'].count()
    complete_timestamps = timestamp_counts[timestamp_counts == config['STOCK_DIM']].index
    df_filtered = df_filtered[df_filtered['date'].isin(complete_timestamps)]
    
    if len(df_filtered) == 0:
        print("⚠️  No complete timestamps found")
        return None
    
    # Create day index
    unique_dates = sorted(df_filtered['date'].unique())
    date_to_day = {date: idx for idx, date in enumerate(unique_dates)}
    df_filtered['day'] = df_filtered['date'].map(date_to_day)
    
    time_span = (df_filtered['date'].max() - df_filtered['date'].min()).total_seconds() / 3600
    print(f"✓ Loaded {len(df_filtered):,} rows ({len(unique_dates)} timestamps, {time_span:.1f}h)")
    
    return df_filtered

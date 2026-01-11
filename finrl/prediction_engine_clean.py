"""
FinRL Prediction Engine - Clean Core for Multi-Agent Systems
"""

import numpy as np


class PredictionEngine:
    """Extract buy/sell/hold signals from FinRL models"""
    
    def __init__(self, actor_model, device, stock_symbols, max_stock=100, action_threshold=10):
        self.act = actor_model
        self.device = device
        self.max_stock = max_stock
        self.action_threshold = action_threshold
        self.stock_symbols = stock_symbols
    
    def get_raw_actions(self, state):
        """Get raw neural network output [-1, 1]"""
        import torch
        with torch.no_grad():
            s_tensor = torch.as_tensor((state,), device=self.device, dtype=torch.float32)
            a_tensor = self.act(s_tensor)
            action = a_tensor.detach().cpu().numpy()[0]
        return action
    
    def get_scaled_actions(self, state):
        """Get scaled actions [-100, 100] for trading"""
        action = self.get_raw_actions(state)
        return (action * self.max_stock).astype(int)
    
    def get_buy_sell_hold_signals(self, state, threshold=None):
        """Parse actions into BUY/SELL/HOLD signals"""
        if threshold is None:
            threshold = self.action_threshold
        
        action = self.get_scaled_actions(state)
        
        buy_indices = np.where(action > threshold)[0]
        sell_indices = np.where(action < -threshold)[0]
        hold_indices = np.where((action >= -threshold) & (action <= threshold))[0]
        
        return {
            'buy': [self.stock_symbols[i] for i in buy_indices],
            'sell': [self.stock_symbols[i] for i in sell_indices],
            'hold': [self.stock_symbols[i] for i in hold_indices],
        }
    
    def get_detailed_predictions(self, state, threshold=None):
        """Get comprehensive prediction output"""
        if threshold is None:
            threshold = self.action_threshold
        
        raw_action = self.get_raw_actions(state)
        scaled_action = self.get_scaled_actions(state)
        signals = self.get_buy_sell_hold_signals(state, threshold)
        
        # Add magnitudes to signals
        buy_signals = [(self.stock_symbols[i], int(scaled_action[i])) 
                      for i in np.where(scaled_action > threshold)[0]]
        sell_signals = [(self.stock_symbols[i], int(scaled_action[i])) 
                       for i in np.where(scaled_action < -threshold)[0]]
        
        return {
            'raw_actions': raw_action.tolist(),
            'scaled_actions': scaled_action.tolist(),
            'signals': {
                'buy': buy_signals,
                'sell': sell_signals,
                'hold': signals['hold'],
            },
            'confidence': float(np.abs(raw_action).max()),
            'action_stats': {
                'mean': float(raw_action.mean()),
                'std': float(raw_action.std()),
                'max': float(raw_action.max()),
                'min': float(raw_action.min()),
                'num_buy': len(buy_signals),
                'num_sell': len(sell_signals),
                'num_hold': len(signals['hold']),
            }
        }


def load_finrl_predictor(model_path, stock_symbols):
    """Load trained FinRL model and create prediction engine"""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = torch.load(model_path, map_location=device)
    return PredictionEngine(actor, device, stock_symbols)
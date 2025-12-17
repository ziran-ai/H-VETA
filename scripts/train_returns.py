"""
Train H-VEDA predicting RETURNS instead of prices
This is the correct approach for financial modeling
"""

import torch
import numpy as np
import os
from config import Config
from data_loader_returns import create_data_loaders
from model import HVEDA_MoE
from train import Trainer
from utils import set_seed

def main():
    print("\n" + "="*80)
    print("H-VEDA Training - PREDICTING RETURNS (Correct Approach)")
    print("="*80)
    
    config = Config()
    config.save_dir = './checkpoints_returns'
    os.makedirs(config.save_dir, exist_ok=True)
    
    set_seed(config.seed)
    
    print(f"\nConfiguration:")
    print(f"  Target: RETURNS (not prices)")
    print(f"  Alpha: {config.alpha}")
    print(f"  Beta: {config.beta}")
    print(f"  Epochs: 50")
    
    # Load data - now predicting returns
    print("\nLoading data (predicting returns)...")
    train_loader, val_loader, test_loader, preprocessor, regime_weights = create_data_loaders(
        config, 
        dev_mode=True
    )
    
    # Add regime weights
    if isinstance(regime_weights, np.ndarray):
        regime_weights = torch.FloatTensor(regime_weights)
    config.regime_weights = regime_weights.to(config.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✅ Regime weights: {regime_weights}")
    
    # Create model
    print("\nInitializing model...")
    model = HVEDA_MoE(config, dev_mode=False)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dev_mode=False
    )
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)
    
    # Train
    history = trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    # Check results
    final_usage = history['expert_usage'][-1]
    print(f"\nFinal Expert Usage:")
    print(f"  Expert 0 (Trend): {final_usage[0]:.2f}%")
    print(f"  Expert 1 (Range): {final_usage[1]:.2f}%")
    print(f"  Expert 2 (Panic): {final_usage[2]:.2f}%")
    
    # Get final validation metrics
    final_val_loss = history['val_loss'][-1]
    final_val_rmse = history['val_rmse'][-1]
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Val Loss: {final_val_loss:.6f}")
    print(f"  Val RMSE: {final_val_rmse:.6f}")
    
    # Calculate baseline for returns
    print(f"\nBaseline for returns prediction:")
    print(f"  Naive (predict 0 return): RMSE ≈ std of returns")
    print(f"  If RMSE < std, model is learning!")

if __name__ == '__main__':
    main()

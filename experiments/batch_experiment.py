"""
Batch experiment script for multi-stock validation
Tests H-VEDA model on multiple stocks to verify generalization
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from config import Config
from data_loader_returns import create_data_loaders
from model import HVEDA_MoE
from train import Trainer
from utils import set_seed

def train_single_stock(stock_path, stock_name, num_epochs=50, save_dir='./checkpoints_batch'):
    """Train model on a single stock"""
    
    print(f"\n{'='*80}")
    print(f"Training: {stock_name}")
    print(f"{'='*80}")
    
    # Create config
    config = Config()
    config.data_path = stock_path
    config.num_epochs = num_epochs
    config.save_dir = os.path.join(save_dir, stock_name)
    os.makedirs(config.save_dir, exist_ok=True)
    
    set_seed(config.seed)
    
    try:
        # Load data
        print(f"Loading data from {stock_path}...")
        train_loader, val_loader, test_loader, preprocessor, regime_weights = create_data_loaders(
            config, 
            dev_mode=False
        )
        
        # Add regime weights
        if isinstance(regime_weights, np.ndarray):
            regime_weights = torch.FloatTensor(regime_weights)
        config.regime_weights = regime_weights.to(config.device if torch.cuda.is_available() else 'cpu')
        
        print(f"Data loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
        
        # Create model
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
        
        # Train
        print(f"Starting training ({num_epochs} epochs)...")
        history = trainer.train()
        
        # Extract final metrics
        final_expert_usage = history['expert_usage'][-1]
        final_val_loss = history['val_loss'][-1]
        
        # Evaluate on test set
        print("Evaluating on test set...")
        model.eval()
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_preds = []
        all_targets = []
        all_gate_weights = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                risk = batch['risk'].to(device)
                
                y_pred, _, gate_weights, _ = model(x, risk)
                
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_gate_weights = np.concatenate(all_gate_weights, axis=0)
        
        # Calculate metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_targets))
        
        pred_direction = (all_preds > 0).astype(int).flatten()
        true_direction = (all_targets > 0).astype(int).flatten()
        dir_acc = (pred_direction == true_direction).mean()
        
        test_expert_usage = all_gate_weights.mean(axis=0) * 100
        
        epsilon = 1e-10
        entropy = -np.sum(all_gate_weights * np.log(all_gate_weights + epsilon), axis=1).mean()
        
        # Calculate baseline
        baseline_rmse = all_targets.std()
        
        results = {
            'stock_name': stock_name,
            'data_path': stock_path,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'num_epochs': num_epochs,
            'final_val_loss': float(final_val_loss),
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'test_directional_accuracy': float(dir_acc),
            'baseline_rmse': float(baseline_rmse),
            'rmse_improvement': float((baseline_rmse - rmse) / baseline_rmse * 100),
            'expert_usage_val': [float(x) for x in final_expert_usage],
            'expert_usage_test': [float(x) for x in test_expert_usage],
            'entropy': float(entropy),
            'regime_weights': [float(x) for x in regime_weights.cpu().numpy()],
            'success': True
        }
        
        print(f"\n{'='*80}")
        print(f"Results for {stock_name}:")
        print(f"  Test RMSE: {rmse:.6f} (Baseline: {baseline_rmse:.6f})")
        print(f"  Directional Accuracy: {dir_acc:.2%}")
        print(f"  Expert Usage (Test): E0={test_expert_usage[0]:.1f}%, E1={test_expert_usage[1]:.1f}%, E2={test_expert_usage[2]:.1f}%")
        print(f"{'='*80}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error training {stock_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'stock_name': stock_name,
            'data_path': stock_path,
            'error': str(e),
            'success': False
        }

def run_batch_experiments(stock_dir='./data/indian_stocks_standard/', 
                         num_epochs=50, 
                         max_stocks=None,
                         output_file='batch_results.json'):
    """Run experiments on multiple stocks"""
    
    print("\n" + "="*80)
    print("H-VEDA BATCH EXPERIMENT - Multi-Stock Validation")
    print("="*80)
    
    # Get all stock files
    stock_files = sorted(glob.glob(os.path.join(stock_dir, '*.csv')))
    
    # Filter out metadata and index files
    stock_files = [f for f in stock_files if 'metadata' not in f.lower() and 'nifty' not in f.lower()]
    
    if max_stocks:
        stock_files = stock_files[:max_stocks]
    
    print(f"\nFound {len(stock_files)} stocks to test")
    print(f"Epochs per stock: {num_epochs}")
    print(f"Estimated time: ~{len(stock_files) * num_epochs * 3 / 60:.1f} minutes\n")
    
    results = []
    
    for i, stock_path in enumerate(stock_files, 1):
        stock_name = os.path.basename(stock_path).replace('.csv', '')
        
        print(f"\n[{i}/{len(stock_files)}] Processing {stock_name}...")
        
        result = train_single_stock(
            stock_path=stock_path,
            stock_name=stock_name,
            num_epochs=num_epochs
        )
        
        results.append(result)
        
        # Save intermediate results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Generate summary
    print("\n" + "="*80)
    print("BATCH EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\nTotal stocks: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        accuracies = [r['test_directional_accuracy'] for r in successful]
        rmses = [r['test_rmse'] for r in successful]
        
        print(f"\nDirectional Accuracy:")
        print(f"  Mean: {np.mean(accuracies):.2%}")
        print(f"  Median: {np.median(accuracies):.2%}")
        print(f"  Std: {np.std(accuracies):.2%}")
        print(f"  Min: {np.min(accuracies):.2%}")
        print(f"  Max: {np.max(accuracies):.2%}")
        print(f"  > 50%: {sum(1 for a in accuracies if a > 0.50)}/{len(accuracies)}")
        print(f"  > 52%: {sum(1 for a in accuracies if a > 0.52)}/{len(accuracies)}")
        print(f"  > 55%: {sum(1 for a in accuracies if a > 0.55)}/{len(accuracies)}")
        
        print(f"\nRMSE:")
        print(f"  Mean: {np.mean(rmses):.6f}")
        print(f"  Median: {np.median(rmses):.6f}")
        
        # Expert usage analysis
        expert_0 = [r['expert_usage_test'][0] for r in successful]
        expert_1 = [r['expert_usage_test'][1] for r in successful]
        expert_2 = [r['expert_usage_test'][2] for r in successful]
        
        print(f"\nExpert Usage (Test Set):")
        print(f"  Expert 0 (Trend): {np.mean(expert_0):.1f}% ± {np.std(expert_0):.1f}%")
        print(f"  Expert 1 (Range): {np.mean(expert_1):.1f}% ± {np.std(expert_1):.1f}%")
        print(f"  Expert 2 (Panic): {np.mean(expert_2):.1f}% ± {np.std(expert_2):.1f}%")
        
        # Check for collapse
        collapsed = sum(1 for r in successful if max(r['expert_usage_test']) > 80)
        print(f"\nExpert Collapse (>80% single expert): {collapsed}/{len(successful)}")
    
    if failed:
        print(f"\nFailed stocks:")
        for r in failed:
            print(f"  - {r['stock_name']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch experiment for multi-stock validation')
    parser.add_argument('--stock_dir', type=str, default='./data/indian_stocks_standard/',
                       help='Directory containing stock CSV files')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs per stock')
    parser.add_argument('--max_stocks', type=int, default=None,
                       help='Maximum number of stocks to test (for quick testing)')
    parser.add_argument('--output', type=str, default='batch_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    results = run_batch_experiments(
        stock_dir=args.stock_dir,
        num_epochs=args.num_epochs,
        max_stocks=args.max_stocks,
        output_file=args.output
    )

if __name__ == '__main__':
    main()

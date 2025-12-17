"""
Batch Testing on Indian Stocks
Tests H-VEDA on 10+ Indian stocks to verify generalization
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from config import Config
from data_loader_returns import create_data_loaders
from model import HVEDA_MoE
from utils import set_seed, save_checkpoint

def test_single_stock(stock_path, stock_name, num_epochs=50):
    """Test model on a single stock"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {stock_name}")
    print(f"{'='*80}")
    
    config = Config()
    config.data_path = stock_path
    config.num_epochs = num_epochs
    config.save_dir = f'./checkpoints_indian/{stock_name}'
    os.makedirs(config.save_dir, exist_ok=True)
    
    set_seed(config.seed)
    
    try:
        # Load data
        train_loader, val_loader, test_loader, preprocessor, regime_weights = create_data_loaders(
            config, dev_mode=False
        )
        
        if isinstance(regime_weights, np.ndarray):
            regime_weights = torch.FloatTensor(regime_weights)
        config.regime_weights = regime_weights.to(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create and train model
        model = HVEDA_MoE(config, dev_mode=False)
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Quick training (fewer epochs for batch testing)
        best_val_loss = float('inf')
        patience = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                risk = batch['risk'].to(device)
                regime = batch['regime'].to(device)
                
                optimizer.zero_grad()
                y_pred, gate_logits, gate_weights, _ = model(x, risk)
                
                # Simple loss
                loss_pred = torch.nn.functional.mse_loss(y_pred, y)
                loss_gate = torch.nn.functional.cross_entropy(gate_logits, regime, weight=config.regime_weights)
                loss = loss_pred + config.alpha * loss_gate
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(device)
                    y = batch['y'].to(device)
                    risk = batch['risk'].to(device)
                    regime = batch['regime'].to(device)
                    
                    y_pred, gate_logits, gate_weights, _ = model(x, risk)
                    loss_pred = torch.nn.functional.mse_loss(y_pred, y)
                    loss_gate = torch.nn.functional.cross_entropy(gate_logits, regime, weight=config.regime_weights)
                    loss = loss_pred + config.alpha * loss_gate
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                save_checkpoint(model, optimizer, epoch, val_loss, 
                              os.path.join(config.save_dir, 'best_model.pt'))
            else:
                patience += 1
            
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.6f}")
        
        # Test evaluation
        model.eval()
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
        
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_gate_weights = np.concatenate(all_gate_weights, axis=0)
        
        # Metrics
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        mae = np.mean(np.abs(all_preds - all_targets))
        
        pred_direction = (all_preds > 0).astype(int)
        true_direction = (all_targets > 0).astype(int)
        dir_acc = (pred_direction == true_direction).mean()
        
        expert_usage = all_gate_weights.mean(axis=0) * 100
        epsilon = 1e-10
        entropy = -np.sum(all_gate_weights * np.log(all_gate_weights + epsilon), axis=1).mean()
        
        baseline_rmse = all_targets.std()
        
        result = {
            'stock_name': stock_name,
            'data_path': stock_path,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'test_directional_accuracy': float(dir_acc),
            'baseline_rmse': float(baseline_rmse),
            'expert_usage': [float(x) for x in expert_usage],
            'entropy': float(entropy),
            'success': True
        }
        
        print(f"\n✓ {stock_name} Results:")
        print(f"  Dir Acc: {dir_acc:.2%}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Expert Usage: {expert_usage[0]:.1f}% / {expert_usage[1]:.1f}% / {expert_usage[2]:.1f}%")
        print(f"  Entropy: {entropy:.3f}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return {
            'stock_name': stock_name,
            'error': str(e),
            'success': False
        }

def main():
    print("\n" + "="*80)
    print("H-VEDA Batch Testing on Indian Stocks")
    print("="*80)
    
    # Select representative stocks
    stocks = [
        'RELIANCE.csv',      # Energy/Conglomerate
        'HDFCBANK.csv',      # Banking
        'TCS.csv',           # IT Services
        'INFY.csv',          # IT Services
        'ICICIBANK.csv',     # Banking
        'HINDUNILVR.csv',    # FMCG
        'BHARTIARTL.csv',    # Telecom
        'ITC.csv',           # FMCG/Tobacco
        'KOTAKBANK.csv',     # Banking
        'SBIN.csv',          # Banking
        'AXISBANK.csv',      # Banking
        'LT.csv',            # Engineering
        'MARUTI.csv',        # Automotive
        'SUNPHARMA.csv',     # Pharma
        'TATAMOTORS.csv'     # Automotive
    ]
    
    stock_dir = './data/indian_stocks_standard/'
    results = []
    
    print(f"\nTesting on {len(stocks)} Indian stocks")
    print(f"Epochs per stock: 50")
    print(f"Estimated time: ~{len(stocks) * 5} minutes\n")
    
    for i, stock_file in enumerate(stocks, 1):
        stock_path = os.path.join(stock_dir, stock_file)
        stock_name = stock_file.replace('.csv', '')
        
        if not os.path.exists(stock_path):
            print(f"[{i}/{len(stocks)}] ✗ {stock_name}: File not found")
            continue
        
        print(f"\n[{i}/{len(stocks)}] Processing {stock_name}...")
        
        result = test_single_stock(stock_path, stock_name, num_epochs=50)
        results.append(result)
        
        # Save intermediate results
        with open('indian_stocks_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("BATCH TESTING SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\nTotal stocks: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        accuracies = [r['test_directional_accuracy'] for r in successful]
        entropies = [r['entropy'] for r in successful]
        
        print(f"\nDirectional Accuracy:")
        print(f"  Mean: {np.mean(accuracies):.2%}")
        print(f"  Median: {np.median(accuracies):.2%}")
        print(f"  Std: {np.std(accuracies):.2%}")
        print(f"  Min: {np.min(accuracies):.2%} ({successful[np.argmin(accuracies)]['stock_name']})")
        print(f"  Max: {np.max(accuracies):.2%} ({successful[np.argmax(accuracies)]['stock_name']})")
        print(f"  > 50%: {sum(1 for a in accuracies if a > 0.50)}/{len(accuracies)}")
        print(f"  > 52%: {sum(1 for a in accuracies if a > 0.52)}/{len(accuracies)}")
        print(f"  > 55%: {sum(1 for a in accuracies if a > 0.55)}/{len(accuracies)}")
        
        print(f"\nExpert Balance (Entropy):")
        print(f"  Mean: {np.mean(entropies):.3f}")
        print(f"  Median: {np.median(entropies):.3f}")
        print(f"  Collapsed (<0.5): {sum(1 for e in entropies if e < 0.5)}/{len(entropies)}")
        
        # Top performers
        print(f"\nTop 5 Performers:")
        sorted_results = sorted(successful, key=lambda x: x['test_directional_accuracy'], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['stock_name']}: {r['test_directional_accuracy']:.2%} (Entropy: {r['entropy']:.3f})")
        
        # Bottom performers
        print(f"\nBottom 5 Performers:")
        for i, r in enumerate(sorted_results[-5:], 1):
            print(f"  {i}. {r['stock_name']}: {r['test_directional_accuracy']:.2%} (Entropy: {r['entropy']:.3f})")
    
    print(f"\nResults saved to: indian_stocks_results.json")
    print("="*80)

if __name__ == '__main__':
    main()

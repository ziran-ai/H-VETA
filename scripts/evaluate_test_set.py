"""
Evaluate trained model on test set
Load best model and compute comprehensive metrics
"""

import torch
import numpy as np
import pandas as pd
from config import Config
from data_loader_returns import create_data_loaders
from model import HVEDA_MoE
from utils import load_checkpoint
from tqdm import tqdm

def evaluate_on_test_set(model, test_loader, config):
    """Evaluate model on test set"""
    model.eval()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_gate_weights = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            risk = batch['risk'].to(device)
            
            # Forward pass
            y_pred, gate_logits, gate_weights, _ = model(x, risk)
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_gate_weights.append(gate_weights.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_gate_weights = np.concatenate(all_gate_weights, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Directional accuracy
    pred_direction = (all_preds > 0).astype(int).flatten()
    true_direction = (all_targets > 0).astype(int).flatten()
    dir_acc = (pred_direction == true_direction).mean()
    
    # Expert usage
    expert_usage = all_gate_weights.mean(axis=0) * 100
    
    # Entropy
    epsilon = 1e-10
    entropy = -np.sum(all_gate_weights * np.log(all_gate_weights + epsilon), axis=1).mean()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': dir_acc,
        'expert_usage': expert_usage,
        'entropy': entropy,
        'predictions': all_preds,
        'targets': all_targets
    }

def calculate_baseline(test_loader):
    """Calculate baseline metrics (predict zero return)"""
    all_targets = []
    
    for batch in test_loader:
        y = batch['y'].numpy()
        all_targets.append(y)
    
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Baseline: predict zero return
    baseline_preds = np.zeros_like(all_targets)
    
    baseline_mse = np.mean((baseline_preds - all_targets) ** 2)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = np.mean(np.abs(baseline_preds - all_targets))
    
    # For directional accuracy, random guessing
    baseline_dir_acc = 0.5
    
    return {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'dir_acc': baseline_dir_acc,
        'std': all_targets.std()
    }

def main():
    print("\n" + "="*80)
    print("TEST SET EVALUATION - Returns Prediction Model")
    print("="*80)
    
    config = Config()
    config.save_dir = './checkpoints_returns'
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, preprocessor, regime_weights = create_data_loaders(
        config, 
        dev_mode=False
    )
    
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Calculate baseline
    print("\nCalculating baseline metrics...")
    baseline = calculate_baseline(test_loader)
    
    print(f"\nBaseline (predict zero return):")
    print(f"  RMSE: {baseline['rmse']:.6f}")
    print(f"  MAE:  {baseline['mae']:.6f}")
    print(f"  Dir Acc: {baseline['dir_acc']:.2%}")
    print(f"  Returns Std: {baseline['std']:.6f}")
    
    # Load model
    print("\nLoading trained model...")
    model = HVEDA_MoE(config, dev_mode=False)
    checkpoint_path = f'{config.save_dir}/best_model.pt'
    
    try:
        optimizer = torch.optim.Adam(model.parameters())
        model, optimizer, epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"✓ Loaded model from epoch {epoch} (val loss: {loss:.6f})")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return
    
    # Evaluate
    results = evaluate_on_test_set(model, test_loader, config)
    
    # Print results
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    
    print(f"\nPerformance Metrics:")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE:  {results['mae']:.6f}")
    print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"  Entropy: {results['entropy']:.4f}")
    
    print(f"\nExpert Usage:")
    print(f"  Expert 0 (Trend): {results['expert_usage'][0]:.2f}%")
    print(f"  Expert 1 (Range): {results['expert_usage'][1]:.2f}%")
    print(f"  Expert 2 (Panic): {results['expert_usage'][2]:.2f}%")
    
    # Compare with baseline
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    rmse_improvement = (baseline['rmse'] - results['rmse']) / baseline['rmse'] * 100
    mae_improvement = (baseline['mae'] - results['mae']) / baseline['mae'] * 100
    dir_improvement = (results['directional_accuracy'] - baseline['dir_acc']) / baseline['dir_acc'] * 100
    
    print(f"\nRMSE:")
    print(f"  Baseline: {baseline['rmse']:.6f}")
    print(f"  Model:    {results['rmse']:.6f}")
    print(f"  Improvement: {rmse_improvement:+.2f}%")
    
    print(f"\nMAE:")
    print(f"  Baseline: {baseline['mae']:.6f}")
    print(f"  Model:    {results['mae']:.6f}")
    print(f"  Improvement: {mae_improvement:+.2f}%")
    
    print(f"\nDirectional Accuracy:")
    print(f"  Baseline: {baseline['dir_acc']:.2%}")
    print(f"  Model:    {results['directional_accuracy']:.2%}")
    print(f"  Improvement: {dir_improvement:+.2f}%")
    
    # Success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    success_count = 0
    
    print(f"\n1. RMSE < Returns Std ({baseline['std']:.6f})")
    if results['rmse'] < baseline['std']:
        print(f"   ✓ PASS: {results['rmse']:.6f} < {baseline['std']:.6f}")
        success_count += 1
    else:
        print(f"   ✗ FAIL: {results['rmse']:.6f} >= {baseline['std']:.6f}")
    
    print(f"\n2. Directional Accuracy > 52%")
    if results['directional_accuracy'] > 0.52:
        print(f"   ✓ PASS: {results['directional_accuracy']:.2%}")
        success_count += 1
    else:
        print(f"   ✗ FAIL: {results['directional_accuracy']:.2%}")
    
    print(f"\n3. Expert Usage Balanced (no expert > 60%)")
    max_usage = results['expert_usage'].max()
    if max_usage < 60:
        print(f"   ✓ PASS: Max usage {max_usage:.2f}%")
        success_count += 1
    else:
        print(f"   ✗ FAIL: Max usage {max_usage:.2f}%")
    
    print(f"\n4. Model beats baseline RMSE")
    if results['rmse'] < baseline['rmse']:
        print(f"   ✓ PASS: {results['rmse']:.6f} < {baseline['rmse']:.6f}")
        success_count += 1
    else:
        print(f"   ✗ FAIL: {results['rmse']:.6f} >= {baseline['rmse']:.6f}")
    
    print(f"\n" + "="*80)
    print(f"OVERALL: {success_count}/4 criteria passed")
    print("="*80)
    
    if success_count >= 3:
        print("\n✅ MODEL IS SUCCESSFUL AND READY FOR DEPLOYMENT")
    elif success_count >= 2:
        print("\n⚠️ MODEL SHOWS PROMISE BUT NEEDS IMPROVEMENT")
    else:
        print("\n❌ MODEL NEEDS SIGNIFICANT IMPROVEMENT")
    
    # Save results
    results_df = pd.DataFrame({
        'predictions': results['predictions'].flatten(),
        'targets': results['targets'].flatten()
    })
    results_df.to_csv(f'{config.save_dir}/test_predictions.csv', index=False)
    print(f"\nPredictions saved to {config.save_dir}/test_predictions.csv")

if __name__ == '__main__':
    main()

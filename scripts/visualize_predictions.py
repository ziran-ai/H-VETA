"""
Visualize predictions to check for lag effect
If predictions are just shifted versions of actual values, directional accuracy will be ~50%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from config import Config
from data_loader_returns import create_data_loaders
from model import HVEDA_MoE
from utils import load_checkpoint

def visualize_lag_effect(model_path, data_path, output_file='lag_analysis.png'):
    """Check if model predictions lag behind actual values"""
    
    print("\n" + "="*80)
    print("LAG EFFECT ANALYSIS")
    print("="*80)
    
    # Load config and data
    config = Config()
    config.data_path = data_path
    
    train_loader, val_loader, test_loader, preprocessor, regime_weights = create_data_loaders(
        config, dev_mode=False
    )
    
    # Load model
    model = HVEDA_MoE(config, dev_mode=False)
    optimizer = torch.optim.Adam(model.parameters())
    
    try:
        model, optimizer, epoch, loss = load_checkpoint(model, optimizer, model_path)
        print(f"✓ Loaded model from epoch {epoch}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Get predictions on test set
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            risk = batch['risk'].to(device)
            
            y_pred, _, _, _ = model(x, risk)
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Calculate correlations with different lags
    print("\nCorrelation Analysis:")
    print(f"  Pred vs Target (lag=0):  {np.corrcoef(preds, targets)[0,1]:.4f}")
    print(f"  Pred vs Target (lag=1):  {np.corrcoef(preds[:-1], targets[1:])[0,1]:.4f}")
    print(f"  Pred vs Target (lag=-1): {np.corrcoef(preds[1:], targets[:-1])[0,1]:.4f}")
    
    # Check if lag=-1 correlation is higher (prediction lags behind)
    corr_0 = np.corrcoef(preds, targets)[0,1]
    corr_lag1 = np.corrcoef(preds[1:], targets[:-1])[0,1]
    
    if corr_lag1 > corr_0:
        print("\n⚠️ WARNING: Model predictions LAG behind actual values!")
        print("   This explains why directional accuracy is ~50%")
        print("   The model is predicting yesterday's movement today")
    
    # Directional accuracy
    pred_dir = (preds > 0).astype(int)
    target_dir = (targets > 0).astype(int)
    dir_acc = (pred_dir == target_dir).mean()
    
    print(f"\nDirectional Accuracy: {dir_acc:.2%}")
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Time series comparison (first 200 points)
    n_plot = min(200, len(preds))
    axes[0].plot(targets[:n_plot], label='Actual Returns', alpha=0.7, linewidth=2)
    axes[0].plot(preds[:n_plot], label='Predicted Returns', alpha=0.7, linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_title('Predicted vs Actual Returns (First 200 samples)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[1].scatter(targets, preds, alpha=0.5, s=20)
    axes[1].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title(f'Scatter Plot (Correlation: {corr_0:.3f})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Return')
    axes[1].set_ylabel('Predicted Return')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Direction confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(target_dir, pred_dir)
    
    axes[2].imshow(cm, cmap='Blues', aspect='auto')
    axes[2].set_title(f'Direction Confusion Matrix (Accuracy: {dir_acc:.2%})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Predicted Direction')
    axes[2].set_ylabel('Actual Direction')
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(['Down', 'Up'])
    axes[2].set_yticklabels(['Down', 'Up'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[2].text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    # Check if model is just predicting mean
    pred_std = preds.std()
    target_std = targets.std()
    
    print(f"\nVariance Analysis:")
    print(f"  Target std:     {target_std:.6f}")
    print(f"  Prediction std: {pred_std:.6f}")
    print(f"  Ratio:          {pred_std/target_std:.2f}")
    
    if pred_std / target_std < 0.5:
        print("\n⚠️ WARNING: Predictions have much lower variance than targets!")
        print("   Model is playing it safe, predicting near-zero returns")
        print("   This is why directional accuracy is ~50% (random)")
    
    # Check directional bias
    pred_up = (preds > 0).sum()
    target_up = (targets > 0).sum()
    
    print(f"\nDirectional Bias:")
    print(f"  Actual up:    {target_up}/{len(targets)} ({target_up/len(targets)*100:.1f}%)")
    print(f"  Predicted up: {pred_up}/{len(preds)} ({pred_up/len(preds)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("\nUse directional loss to force model to commit to predictions!")
    print("Run: python train_directional.py")
    print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints_returns/best_model.pt')
    parser.add_argument('--data', type=str, default='./data/googledata/GOOGL.csv')
    parser.add_argument('--output', type=str, default='lag_analysis.png')
    
    args = parser.parse_args()
    
    visualize_lag_effect(args.model, args.data, args.output)

if __name__ == '__main__':
    main()

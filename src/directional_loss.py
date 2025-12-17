"""
Directional Loss Functions for Financial Time Series
Focus on sign prediction rather than just MSE minimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalLoss(nn.Module):
    """
    Penalizes predictions that get the direction (sign) wrong
    Uses smooth approximation of sign function for differentiability
    """
    
    def __init__(self, temperature=10.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred, target):
        """
        Args:
            pred: predicted returns (batch_size, 1)
            target: actual returns (batch_size, 1)
        
        Returns:
            directional_loss: penalty for wrong direction
        """
        # Use tanh as smooth approximation of sign
        # tanh(x * temperature) ≈ sign(x) when temperature is large
        pred_sign = torch.tanh(pred * self.temperature)
        target_sign = torch.tanh(target * self.temperature)
        
        # Directional loss: 1 - cosine similarity of signs
        # When signs match: loss ≈ 0
        # When signs opposite: loss ≈ 2
        directional_loss = 1 - (pred_sign * target_sign).mean()
        
        return directional_loss


class DirectionalAccuracyLoss(nn.Module):
    """
    Binary cross-entropy on direction classification
    Treats direction prediction as a classification problem
    """
    
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        """
        Convert to binary classification: up (1) or down (0)
        """
        # Convert to binary labels
        target_direction = (target > self.threshold).float()
        
        # Use raw predictions as logits
        loss = self.bce(pred, target_direction)
        
        return loss


class CombinedFinancialLoss(nn.Module):
    """
    Combined loss: MSE for magnitude + Directional for sign
    L_total = alpha * L_MSE + beta * L_Direction + gamma * L_Gate + delta * L_Balance
    """
    
    def __init__(self, 
                 mse_weight=1.0,
                 direction_weight=2.0,  # Higher weight on direction!
                 gate_weight=1.0,
                 balance_weight=0.1,
                 regime_weights=None,
                 temperature=10.0,
                 dev_mode=False):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.gate_weight = gate_weight
        self.balance_weight = balance_weight
        self.dev_mode = dev_mode
        
        # Loss components
        self.mse = nn.MSELoss()
        self.directional = DirectionalLoss(temperature=temperature)
        
        if regime_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=regime_weights)
        else:
            self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target, gate_logits, regime_labels, gate_weights):
        """
        Compute combined loss with emphasis on directional accuracy
        
        Args:
            pred: predicted returns
            target: actual returns
            gate_logits: raw gate outputs
            regime_labels: ground truth regime labels
            gate_weights: softmax gate weights
        
        Returns:
            total_loss, loss_dict
        """
        # 1. MSE Loss (magnitude)
        mse_loss = self.mse(pred, target)
        
        # 2. Directional Loss (sign) - KEY ADDITION!
        direction_loss = self.directional(pred, target)
        
        # 3. Gate supervision loss
        gate_loss = self.ce(gate_logits, regime_labels)
        
        # 4. Load balancing loss
        num_experts = gate_weights.size(1)
        expert_usage = gate_weights.mean(dim=0)
        target_usage = 1.0 / num_experts
        balance_loss = -((expert_usage - target_usage) ** 2).sum()
        
        # Combined loss with directional emphasis
        total_loss = (
            self.mse_weight * mse_loss +
            self.direction_weight * direction_loss +  # Higher weight!
            self.gate_weight * gate_loss +
            self.balance_weight * balance_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'gate': gate_loss.item(),
            'balance': balance_loss.item()
        }
        
        if self.dev_mode:
            # Calculate actual directional accuracy for monitoring
            pred_sign = (pred > 0).float()
            target_sign = (target > 0).float()
            dir_acc = (pred_sign == target_sign).float().mean()
            loss_dict['dir_accuracy'] = dir_acc.item()
        
        return total_loss, loss_dict


class ThresholdedDirectionalLoss(nn.Module):
    """
    Only penalize direction errors when magnitude is significant
    Filters out noise from small fluctuations
    """
    
    def __init__(self, threshold=0.005, temperature=10.0):
        """
        Args:
            threshold: minimum return magnitude to consider (e.g., 0.5%)
            temperature: smoothness of sign approximation
        """
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        
    def forward(self, pred, target):
        """
        Only compute directional loss for significant moves
        """
        # Mask for significant moves
        significant = (torch.abs(target) > self.threshold).float()
        
        if significant.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Smooth sign approximation
        pred_sign = torch.tanh(pred * self.temperature)
        target_sign = torch.tanh(target * self.temperature)
        
        # Directional loss only on significant moves
        directional_error = 1 - (pred_sign * target_sign)
        weighted_error = (directional_error * significant).sum() / significant.sum()
        
        return weighted_error


def test_directional_loss():
    """Test directional loss functions"""
    print("\n" + "="*80)
    print("Testing Directional Loss Functions")
    print("="*80)
    
    # Create test data
    pred = torch.tensor([[0.01], [-0.02], [0.03], [-0.01], [0.005]])
    target = torch.tensor([[0.015], [-0.015], [-0.02], [0.02], [0.001]])
    
    print("\nTest Data:")
    print("Predictions:", pred.squeeze().tolist())
    print("Targets:    ", target.squeeze().tolist())
    
    # Test DirectionalLoss
    dir_loss = DirectionalLoss(temperature=10.0)
    loss = dir_loss(pred, target)
    print(f"\nDirectional Loss: {loss.item():.4f}")
    
    # Calculate actual accuracy
    pred_sign = (pred > 0).float()
    target_sign = (target > 0).float()
    accuracy = (pred_sign == target_sign).float().mean()
    print(f"Directional Accuracy: {accuracy.item():.2%}")
    
    # Test ThresholdedDirectionalLoss
    thresh_loss = ThresholdedDirectionalLoss(threshold=0.01, temperature=10.0)
    loss_thresh = thresh_loss(pred, target)
    print(f"\nThresholded Directional Loss (>1%): {loss_thresh.item():.4f}")
    
    print("\n" + "="*80)
    print("Key Insight:")
    print("  - Directional loss penalizes wrong sign predictions")
    print("  - Higher loss = more direction errors")
    print("  - Thresholded version filters out noise")
    print("="*80)

if __name__ == '__main__':
    test_directional_loss()

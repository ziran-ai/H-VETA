"""
H-VEDA (Rg-MoE) model architecture
Regime-Gated Mixture of Experts with Volatility-Enhanced Decision-making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ExpertBlock, GateNetwork
from utils import calculate_entropy


class HVEDA_MoE(nn.Module):
    """
    H-VEDA: Regime-Gated Mixture of Experts model
    """
    
    def __init__(self, config, dev_mode=False):
        """
        Args:
            config: configuration object
            dev_mode: whether to print debug information
        """
        super().__init__()
        self.config = config
        self.dev_mode = dev_mode
        self.num_experts = config.num_experts
        self.entropy_threshold = config.entropy_threshold
        
        # Expert pool with different risk sensitivities
        self.experts = nn.ModuleList([
            ExpertBlock(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                risk_lambda=config.expert_lambdas[i],
                num_heads=config.num_heads,
                dropout=0.1,
                dev_mode=dev_mode
            )
            for i in range(config.num_experts)
        ])
        
        # Gate network with noise injection
        gate_input_dim = config.input_dim + 1  # PCA features + risk
        gate_noise_std = config.gate_noise_std if hasattr(config, 'gate_noise_std') else 0.1
        self.gate = GateNetwork(
            input_dim=gate_input_dim,
            num_experts=config.num_experts,
            hidden_dim=64,
            dropout=0.1,
            noise_std=gate_noise_std,
            dev_mode=dev_mode
        )
        
        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        if self.dev_mode:
            print(f"[Model] Initialized H-VEDA with {self.num_experts} experts")
            print(f"[Model] Expert lambdas: {config.expert_lambdas}")
            print(f"[Model] Hidden dim: {config.hidden_dim}")
    
    def forward(self, x_trend, risk_sigma):
        """
        Forward pass through the model
        
        Args:
            x_trend: (batch_size, seq_len, input_dim) - PCA features
            risk_sigma: (batch_size, 1) - volatility/risk measure
        
        Returns:
            prediction: (batch_size, 1) - predicted value
            gate_logits: (batch_size, num_experts) - gate logits for loss
            gate_weights: (batch_size, num_experts) - expert weights
            expert_outputs: (batch_size, num_experts, hidden_dim) - individual expert outputs
        """
        batch_size = x_trend.size(0)
        
        if self.dev_mode:
            print(f"\n[Model Forward] Batch size: {batch_size}")
            print(f"[Model Forward] Input trend shape: {x_trend.shape}")
            print(f"[Model Forward] Risk sigma shape: {risk_sigma.shape}")
        
        # Compute expert outputs in parallel
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            if self.dev_mode:
                print(f"\n[Model Forward] Processing Expert {i}")
            out = expert(x_trend, risk_sigma)
            expert_outputs.append(out)
        
        # Stack expert outputs: (batch_size, num_experts, hidden_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        if self.dev_mode:
            print(f"[Model Forward] Stacked expert outputs shape: {expert_outputs.shape}")
        
        # Compute gate weights
        # Use last timestep of trend features + risk for routing decision
        last_step_trend = x_trend[:, -1, :]  # (batch_size, input_dim)
        gate_input = torch.cat([last_step_trend, risk_sigma], dim=-1)
        
        if self.dev_mode:
            print(f"[Model Forward] Gate input shape: {gate_input.shape}")
        
        gate_logits, gate_weights = self.gate(gate_input)
        
        # Weighted fusion of expert outputs
        # gate_weights: (batch_size, num_experts) -> (batch_size, num_experts, 1)
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        
        # Weighted sum: (batch_size, num_experts, hidden_dim) * (batch_size, num_experts, 1)
        # -> sum over experts -> (batch_size, hidden_dim)
        context_vector = torch.sum(expert_outputs * gate_weights_expanded, dim=1)
        
        if self.dev_mode:
            print(f"[Model Forward] Context vector shape: {context_vector.shape}")
        
        # Final prediction
        prediction = self.predictor(context_vector)
        
        if self.dev_mode:
            print(f"[Model Forward] Prediction shape: {prediction.shape}")
            print(f"[Model Forward] Prediction range: [{prediction.min().item():.4f}, {prediction.max().item():.4f}]")
        
        return prediction, gate_logits, gate_weights, expert_outputs
    
    def predict_with_notrade(self, x_trend, risk_sigma):
        """
        Make prediction with no-trade decision logic
        
        Args:
            x_trend: (batch_size, seq_len, input_dim)
            risk_sigma: (batch_size, 1)
        
        Returns:
            prediction: (batch_size, 1) - predicted value (or None for no-trade)
            gate_weights: (batch_size, num_experts) - expert weights
            entropy: (batch_size,) - entropy of gate distribution
            should_trade: (batch_size,) - boolean mask for trading decision
        """
        with torch.no_grad():
            prediction, gate_logits, gate_weights, _ = self.forward(x_trend, risk_sigma)
            
            # Calculate entropy of gate distribution
            entropy = calculate_entropy(gate_weights)
            
            # No-trade decision: high entropy means uncertain regime
            should_trade = entropy < self.entropy_threshold
            
            if self.dev_mode:
                print(f"[No-Trade Logic] Entropy: {entropy.mean().item():.4f}")
                print(f"[No-Trade Logic] Should trade ratio: {should_trade.float().mean().item():.2%}")
            
            return prediction, gate_weights, entropy, should_trade
    
    def get_expert_contributions(self, x_trend, risk_sigma):
        """
        Get individual expert predictions (for analysis)
        
        Args:
            x_trend: (batch_size, seq_len, input_dim)
            risk_sigma: (batch_size, 1)
        
        Returns:
            expert_predictions: (batch_size, num_experts) - individual expert predictions
            gate_weights: (batch_size, num_experts) - expert weights
        """
        with torch.no_grad():
            batch_size = x_trend.size(0)
            
            # Get expert outputs
            expert_outputs = []
            for expert in self.experts:
                out = expert(x_trend, risk_sigma)
                expert_outputs.append(out)
            
            expert_outputs = torch.stack(expert_outputs, dim=1)
            
            # Get individual predictions from each expert
            expert_predictions = []
            for i in range(self.num_experts):
                pred = self.predictor(expert_outputs[:, i, :])
                expert_predictions.append(pred)
            
            expert_predictions = torch.cat(expert_predictions, dim=1)
            
            # Get gate weights
            last_step_trend = x_trend[:, -1, :]
            gate_input = torch.cat([last_step_trend, risk_sigma], dim=-1)
            _, gate_weights = self.gate(gate_input)
            
            return expert_predictions, gate_weights


class HVEDA_Loss(nn.Module):
    """
    Joint loss function for H-VEDA
    Combines prediction loss with gate supervision loss and load balancing
    """
    
    def __init__(self, alpha=0.1, beta=0.01, regime_weights=None, dev_mode=False):
        """
        Args:
            alpha: weight for gate supervision loss
            beta: weight for load balancing loss
            regime_weights: class weights for imbalanced regimes (CRITICAL FIX)
            dev_mode: whether to print debug information
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dev_mode = dev_mode
        self.mse = nn.MSELoss()
        
        # FIXED: Weighted CrossEntropyLoss to handle regime imbalance
        if regime_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=regime_weights)
            if dev_mode:
                print(f"[Loss] Using weighted CE with weights: {regime_weights}")
        else:
            self.ce = nn.CrossEntropyLoss()
            if dev_mode:
                print("[Loss] WARNING: No regime weights provided, using unweighted CE")
    
    def compute_load_balancing_loss(self, gate_weights):
        """
        Compute load balancing loss to prevent expert collapse
        Encourages uniform distribution of samples across experts
        
        Args:
            gate_weights: (batch_size, num_experts) - softmax probabilities
        
        Returns:
            load_balance_loss: scalar loss value
        """
        # Average gate weights across batch
        avg_weights = gate_weights.mean(dim=0)  # (num_experts,)
        
        # Compute coefficient of variation (CV)
        mean_weight = avg_weights.mean()
        std_weight = avg_weights.std()
        cv = std_weight / (mean_weight + 1e-10)
        
        # Entropy-based loss (higher entropy = more balanced)
        entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-10))
        max_entropy = torch.log(torch.tensor(len(avg_weights), dtype=torch.float32, device=gate_weights.device))
        
        # Combine both metrics
        load_balance_loss = cv - (entropy / max_entropy)
        
        return load_balance_loss
    
    def forward(self, y_pred, y_true, gate_logits, regime_labels, gate_weights=None):
        """
        Compute joint loss
        
        Args:
            y_pred: (batch_size, 1) - predicted values
            y_true: (batch_size, 1) - true values
            gate_logits: (batch_size, num_experts) - gate logits
            regime_labels: (batch_size,) - regime labels (0, 1, 2)
            gate_weights: (batch_size, num_experts) - gate softmax weights (optional)
        
        Returns:
            total_loss: combined loss
            loss_prediction: MSE loss
            loss_gate: cross-entropy loss for gate
            loss_balance: load balancing loss
        """
        # Prediction loss
        loss_prediction = self.mse(y_pred, y_true)
        
        # Gate supervision loss
        loss_gate = self.ce(gate_logits, regime_labels)
        
        # Load balancing loss
        if gate_weights is not None and self.beta > 0:
            loss_balance = self.compute_load_balancing_loss(gate_weights)
        else:
            loss_balance = torch.tensor(0.0, device=y_pred.device)
        
        # Total loss
        total_loss = loss_prediction + (self.alpha * loss_gate) + (self.beta * loss_balance)
        
        if self.dev_mode:
            print(f"[Loss] Prediction loss: {loss_prediction.item():.6f}")
            print(f"[Loss] Gate loss: {loss_gate.item():.6f}")
            print(f"[Loss] Balance loss: {loss_balance.item():.6f}")
            print(f"[Loss] Total loss: {total_loss.item():.6f}")
        
        return total_loss, loss_prediction, loss_gate, loss_balance
    
    def update_alpha(self, new_alpha):
        """Update alpha value (for decay schedule)"""
        self.alpha = new_alpha
    
    def update_beta(self, new_beta):
        """Update beta value (for load balancing weight)"""
        self.beta = new_beta

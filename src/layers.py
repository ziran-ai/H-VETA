"""
Core neural network layers for H-VEDA (Rg-MoE)
Includes Volatility-Modulated Attention (VMA) and Expert blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VolatilityModulatedAttention(nn.Module):
    """
    Volatility-Modulated Attention (VMA) mechanism
    Penalizes distant information based on market risk
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, dev_mode=False):
        """
        Args:
            hidden_dim: dimension of hidden states
            num_heads: number of attention heads
            dropout: dropout rate
            dev_mode: whether to print debug information
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dev_mode = dev_mode
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, Q, K, V, risk_sigma, risk_lambda):
        """
        Forward pass with volatility-modulated attention
        Normalized volatility to prevent attention vanishing
        
        Args:
            Q, K, V: (batch_size, seq_len, hidden_dim)
            risk_sigma: (batch_size, 1) - current market volatility
            risk_lambda: scalar - risk aversion coefficient
        
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = Q.shape
        
        if self.dev_mode:
            print(f"[VMA] Input shape: {Q.shape}")
            print(f"[VMA] Risk sigma shape: {risk_sigma.shape}")
            print(f"[VMA] Lambda penalty: {risk_lambda}")
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.dev_mode:
            print(f"[VMA] Q shape after split: {Q.shape}")
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.dev_mode:
            print(f"[VMA] Attention scores shape: {scores.shape}")
        
        # Compute temporal distance penalty
        positions = torch.arange(seq_len, device=Q.device).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        
        # Normalize distance by sequence length to prevent explosion
        normalized_distance = distance_matrix / seq_len
        distance_sq = normalized_distance ** 2
        
        # Clip and normalize volatility to prevent huge penalties
        # Typical volatility range: 0.01-0.10, normalize to 0-1
        risk_sigma_clipped = torch.clamp(risk_sigma, min=0.001, max=0.2)
        risk_sigma_normalized = (risk_sigma_clipped - 0.001) / (0.2 - 0.001)
        
        # Scale lambda to reasonable range (0.1-1.0 instead of 0.5-10)
        lambda_scaled = risk_lambda / 10.0
        
        # Risk-based penalty: scaled_λ * normalized_σ * normalized_distance²
        risk_term = lambda_scaled * risk_sigma_normalized.unsqueeze(-1).unsqueeze(-1) * distance_sq.unsqueeze(0).unsqueeze(0)
        
        if self.dev_mode:
            print(f"[VMA] Risk term shape: {risk_term.shape}")
            print(f"[VMA] Risk term range: [{risk_term.min().item():.4f}, {risk_term.max().item():.4f}]")
        
        # Apply penalty to attention scores
        modified_scores = scores - risk_term
        
        if self.dev_mode:
            print(f"[VMA] Modified scores range: [{modified_scores.min().item():.4f}, {modified_scores.max().item():.4f}]")
        
        # Softmax to get attention weights
        attn_weights = F.softmax(modified_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Final linear projection
        output = self.out_proj(context)
        
        if self.dev_mode:
            print(f"[VMA] Output shape: {output.shape}")
        
        return output


class ExpertBlock(nn.Module):
    """
    Expert block with LSTM and VMA
    Each expert has a unique risk sensitivity (lambda)
    """
    
    def __init__(self, input_dim, hidden_dim, risk_lambda, num_heads=4, dropout=0.1, dev_mode=False):
        """
        Args:
            input_dim: dimension of input features
            hidden_dim: dimension of hidden states
            risk_lambda: risk aversion coefficient for this expert
            num_heads: number of attention heads
            dropout: dropout rate
            dev_mode: whether to print debug information
        """
        super().__init__()
        self.risk_lambda = risk_lambda
        self.hidden_dim = hidden_dim
        self.dev_mode = dev_mode
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        self.vma = VolatilityModulatedAttention(
            hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            dev_mode=dev_mode
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, risk_sigma):
        """
        Forward pass through expert
        
        Args:
            x: (batch_size, seq_len, input_dim)
            risk_sigma: (batch_size, 1)
        
        Returns:
            output: (batch_size, hidden_dim) - last timestep features
        """
        if self.dev_mode:
            print(f"[Expert λ={self.risk_lambda}] Input shape: {x.shape}")
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.dev_mode:
            print(f"[Expert λ={self.risk_lambda}] LSTM output shape: {lstm_out.shape}")
        
        # Apply VMA
        attn_out = self.vma(lstm_out, lstm_out, lstm_out, risk_sigma, self.risk_lambda)
        
        # Feed-forward network
        ff_out = self.feed_forward(attn_out)
        output = self.layer_norm(attn_out + ff_out)
        
        # Extract last timestep
        final_output = output[:, -1, :]
        
        if self.dev_mode:
            print(f"[Expert λ={self.risk_lambda}] Final output shape: {final_output.shape}")
        
        return final_output


class GateNetwork(nn.Module):
    """
    Gate network for regime-based expert routing with noise injection
    """
    
    def __init__(self, input_dim, num_experts, hidden_dim=64, dropout=0.1, noise_std=0.0, dev_mode=False):
        """
        Args:
            input_dim: dimension of input features (trend + risk)
            num_experts: number of experts
            hidden_dim: hidden dimension for gate MLP
            dropout: dropout rate
            noise_std: standard deviation of exploration noise
            dev_mode: whether to print debug information
        """
        super().__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.dev_mode = dev_mode
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, x, add_noise=True):
        """
        Forward pass through gate with optional noise injection
        
        Args:
            x: (batch_size, input_dim)
            add_noise: whether to add exploration noise (for training)
        
        Returns:
            logits: (batch_size, num_experts) - raw logits for loss computation
            weights: (batch_size, num_experts) - softmax probabilities for routing
        """
        if self.dev_mode:
            print(f"[Gate] Input shape: {x.shape}")
        
        logits = self.gate(x)
        
        # Add Gumbel noise for exploration during training
        if add_noise and self.noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            if self.dev_mode:
                print(f"[Gate] Added noise with std={self.noise_std}")
        
        weights = F.softmax(logits, dim=1)
        
        if self.dev_mode:
            print(f"[Gate] Logits shape: {logits.shape}")
            print(f"[Gate] Weights (first sample): {weights[0].detach().cpu().numpy()}")
        
        return logits, weights
    
    def set_noise_std(self, noise_std):
        """Update noise standard deviation"""
        self.noise_std = noise_std

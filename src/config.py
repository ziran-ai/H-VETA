"""
Configuration file for H-VEDA (Rg-MoE) model
"""

class Config:
    # Data parameters
    data_path = './data/googledata/GOOGL.csv'
    sequence_length = 60
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # CEEMD parameters
    ceemd_trials = 100
    ceemd_noise_std = 0.2
    ceemd_max_imf = 10
    
    # PCA parameters
    pca_components = 10
    pca_variance_threshold = 0.95
    
    # Regime labeling parameters
    regime_window = 20
    trend_r2_threshold = 0.8
    high_volatility_multiplier = 1.5
    
    # Model architecture parameters
    input_dim = 10  # PCA components
    hidden_dim = 128
    num_experts = 3
    num_heads = 4
    
    # Expert risk sensitivity (lambda values)
    expert_lambdas = [0.5, 2.0, 10.0]  # Trend, Range, Panic
    
    # Training parameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-5
    alpha = 5.0  # Joint loss weight for gate supervision (increased to prevent collapse)
    alpha_decay = 0.995  # Decay alpha over epochs (slower decay)
    beta = 1.0  # Load balancing loss weight
    gate_noise_std = 0.1  # Exploration noise for gate network
    
    # Expert pre-training
    use_pretrain = False  # Whether to use expert pre-training
    pretrain_epochs = 10  # Number of epochs for pre-training each expert
    
    # Early stopping
    patience = 15
    min_delta = 1e-4
    
    # No-trade decision parameters
    entropy_threshold = 0.8
    
    # Device
    device = 'cuda'
    
    # Random seed
    seed = 42
    
    # Logging
    log_interval = 10
    save_dir = './checkpoints'
    
    # Development mode
    dev_mode = False

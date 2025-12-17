"""
Create Beautiful Publication-Ready Figures
Professional color schemes and modern styling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# 设置专业的样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

# 专业配色方案
COLORS = {
    'primary': '#2E86AB',      # 深蓝色
    'secondary': '#A23B72',    # 紫红色
    'accent': '#F18F01',       # 橙色
    'success': '#06A77D',      # 绿色
    'warning': '#F77F00',      # 深橙色
    'danger': '#D62828',       # 红色
    'neutral': '#6C757D',      # 灰色
    'light': '#E9ECEF',        # 浅灰色
}

# 专家配色（更柔和的渐变）
EXPERT_COLORS = ['#4A90E2', '#7B68EE', '#FF6B6B']  # 蓝、紫、红
REGIME_COLORS = ['#5DADE2', '#58D68D', '#EC7063']  # 浅蓝、绿、红

def setup_figure_style():
    """设置全局图表样式"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#666666',
        'ytick.color': '#666666',
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })

def fig1_performance_comparison():
    """Figure 1: 性能对比 - 美化版"""
    setup_figure_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = ['LSTM', 'Transformer', 'MoE-Random', 'MoE-Learned', 'H-VEDA\n(Ours)']
    accuracies = [52.1, 53.4, 51.8, 54.2, 55.6]
    
    # 使用渐变色
    colors = [COLORS['neutral']] * 4 + [COLORS['primary']]
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.85, 
                  edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#333333')
    
    # 添加基线
    ax.axhline(y=50, color=COLORS['danger'], linestyle='--', 
              linewidth=2, alpha=0.7, label='Random Baseline (50%)')
    ax.axhline(y=55, color=COLORS['success'], linestyle='--', 
              linewidth=2, alpha=0.5, label='Strong Performance (55%)')
    
    ax.set_ylabel('Directional Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([48, 58])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/fig1_performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 1: Performance comparison (美化版)")

def fig2_expert_usage():
    """Figure 2: 专家使用率 - 美化版"""
    setup_figure_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    regimes = ['Trend', 'Range', 'Panic']
    expert_0 = [46, 28, 26]
    expert_1 = [25, 52, 23]
    expert_2 = [18, 22, 60]
    
    x = np.arange(len(regimes))
    width = 0.25
    
    # 柱状图
    bars1 = axes[0].bar(x - width, expert_0, width, label='Expert 0 (Trend)', 
                       color=EXPERT_COLORS[0], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = axes[0].bar(x, expert_1, width, label='Expert 1 (Range)', 
                       color=EXPERT_COLORS[1], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars3 = axes[0].bar(x + width, expert_2, width, label='Expert 2 (Panic)', 
                       color=EXPERT_COLORS[2], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}%', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
    
    axes[0].set_ylabel('Activation Percentage (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Expert Activation by Market Regime', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regimes)
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # 热图
    data = np.array([expert_0, expert_1, expert_2])
    im = axes[1].imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=70)
    
    axes[1].set_xticks(np.arange(len(regimes)))
    axes[1].set_yticks(np.arange(3))
    axes[1].set_xticklabels(regimes)
    axes[1].set_yticklabels(['Expert 0', 'Expert 1', 'Expert 2'])
    axes[1].set_title('Expert Specialization Heatmap', fontsize=14, fontweight='bold')
    
    # 添加文本标注
    for i in range(3):
        for j in range(3):
            text = axes[1].text(j, i, f'{data[i, j]:.0f}%',
                              ha="center", va="center", color="white" if data[i, j] > 40 else "black", 
                              fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=axes[1], label='Activation %')
    cbar.outline.set_linewidth(0)
    
    plt.tight_layout()
    plt.savefig('figures/fig2_expert_usage.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 2: Expert usage (美化版)")

def fig3_data_leakage():
    """Figure 3: 数据泄露影响 - 美化版"""
    setup_figure_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['With\nLeakage', 'Without\nLeakage']
    val_acc = [58.2, 56.1]
    test_acc = [61.2, 55.6]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # 柱状图
    bars1 = axes[0].bar(x - width/2, val_acc, width, label='Validation', 
                       color=COLORS['secondary'], alpha=0.85, edgecolor='white', linewidth=2)
    bars2 = axes[0].bar(x + width/2, test_acc, width, label='Test', 
                       color=COLORS['primary'], alpha=0.85, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')
    
    axes[0].axhline(y=50, color=COLORS['danger'], linestyle='--', 
                   linewidth=2, alpha=0.6, label='Random')
    axes[0].set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Impact of Data Leakage', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].set_ylim([45, 65])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Gap 分析
    gaps = [test_acc[i] - val_acc[i] for i in range(len(methods))]
    colors_gap = [COLORS['danger'] if g < 0 else COLORS['success'] for g in gaps]
    
    bars = axes[1].bar(methods, gaps, color=colors_gap, alpha=0.85, 
                      edgecolor='white', linewidth=2)
    axes[1].axhline(y=0, color='#333333', linestyle='-', linewidth=1.5)
    
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., 
                    height + (0.3 if height > 0 else -0.5),
                    f'{gap:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')
    
    axes[1].set_ylabel('Test - Val Gap (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation-Test Gap\n(Negative = Suspicious)', 
                     fontsize=14, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/fig3_data_leakage.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 3: Data leakage (美化版)")

def fig4_ablation_study():
    """Figure 4: 消融实验 - 美化版"""
    setup_figure_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = ['Full\nModel', 'w/o\nRegime\nGate', 'w/o\nVMA', 'w/o\nPre-train', 'w/o\nWeighted\nLoss']
    dir_acc = [55.6, 53.1, 54.3, 52.8, 53.5]
    entropy = [1.09, 0.42, 1.05, 0.38, 0.51]
    
    # 准确率
    colors_acc = [COLORS['success'] if acc >= 55 else COLORS['warning'] if acc >= 53 else COLORS['danger'] 
                  for acc in dir_acc]
    colors_acc[0] = COLORS['primary']  # 完整模型用主色
    
    bars = axes[0].bar(variants, dir_acc, color=colors_acc, alpha=0.85, 
                      edgecolor='white', linewidth=2)
    
    for bar, acc in zip(bars, dir_acc):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    axes[0].axhline(y=50, color=COLORS['danger'], linestyle='--', 
                   linewidth=2, alpha=0.5, label='Random')
    axes[0].set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Ablation Study: Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=True)
    axes[0].set_ylim([48, 58])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # 熵值
    colors_ent = [COLORS['success'] if e >= 1.0 else COLORS['warning'] if e >= 0.5 else COLORS['danger'] 
                  for e in entropy]
    colors_ent[0] = COLORS['primary']
    
    bars = axes[1].bar(variants, entropy, color=colors_ent, alpha=0.85, 
                      edgecolor='white', linewidth=2)
    
    for bar, e in zip(bars, entropy):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{e:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    axes[1].axhline(y=1.10, color=COLORS['success'], linestyle='--', 
                   linewidth=2, alpha=0.5, label='Max Entropy')
    axes[1].axhline(y=0.5, color=COLORS['danger'], linestyle='--', 
                   linewidth=2, alpha=0.5, label='Collapse')
    axes[1].set_ylabel('Expert Usage Entropy', fontsize=12, fontweight='bold')
    axes[1].set_title('Ablation Study: Expert Balance', fontsize=14, fontweight='bold')
    axes[1].legend(frameon=True, fancybox=True, fontsize=9)
    axes[1].set_ylim([0, 1.3])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/fig4_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 4: Ablation study (美化版)")

def fig5_cross_market():
    """Figure 5: 跨市场对比 - 美化版"""
    setup_figure_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    markets = ['GOOGL\n(US)', 'RELIANCE\n(India)', 'HDFCBANK\n(India)', 'TCS\n(India)']
    accuracies = [55.6, 50.7, 49.3, 48.4]
    entropies = [1.09, 0.69, 0.68, 0.72]
    
    # 准确率
    colors_acc = [COLORS['primary']] + [COLORS['secondary']] * 3
    bars = axes[0].bar(markets, accuracies, color=colors_acc, alpha=0.85, 
                      edgecolor='white', linewidth=2)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    axes[0].axhline(y=50, color=COLORS['danger'], linestyle='--', 
                   linewidth=2, alpha=0.6, label='Random')
    axes[0].axhline(y=55, color=COLORS['success'], linestyle='--', 
                   linewidth=2, alpha=0.4, label='Strong')
    axes[0].set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Cross-Market Generalization', fontsize=14, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=True)
    axes[0].set_ylim([45, 60])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # 熵值
    colors_ent = [COLORS['primary']] + [COLORS['accent']] * 3
    bars = axes[1].bar(markets, entropies, color=colors_ent, alpha=0.85, 
                      edgecolor='white', linewidth=2)
    
    for bar, ent in zip(bars, entropies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{ent:.2f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    axes[1].axhline(y=1.10, color=COLORS['success'], linestyle='--', 
                   linewidth=2, alpha=0.5, label='Max')
    axes[1].axhline(y=0.5, color=COLORS['danger'], linestyle='--', 
                   linewidth=2, alpha=0.5, label='Collapse')
    axes[1].set_ylabel('Expert Usage Entropy', fontsize=12, fontweight='bold')
    axes[1].set_title('Expert Balance Across Markets', fontsize=14, fontweight='bold')
    axes[1].legend(frameon=True, fancybox=True)
    axes[1].set_ylim([0, 1.3])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/fig5_cross_market.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 5: Cross-market (美化版)")

def fig6_confidence_intervals():
    """Figure 6: 置信区间 - 美化版"""
    setup_figure_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['LSTM', 'Transformer', 'MoE-Learned', 'H-VEDA']
    accuracies = [52.1, 53.4, 54.2, 55.6]
    ci_lower = [48.1, 49.3, 50.2, 51.8]
    ci_upper = [56.1, 57.5, 58.2, 59.4]
    
    y_pos = np.arange(len(models))
    
    colors = [COLORS['neutral']] * 3 + [COLORS['primary']]
    
    for i, (model, acc, lower, upper, color) in enumerate(zip(models, accuracies, ci_lower, ci_upper, colors)):
        ax.errorbar(acc, i, xerr=[[acc-lower], [upper-acc]], 
                   fmt='o', markersize=12, capsize=8, capthick=2.5,
                   color=color, ecolor=color, linewidth=2.5, alpha=0.85,
                   markeredgecolor='white', markeredgewidth=2)
        
        # 添加数值标签
        ax.text(acc, i + 0.25, f'{acc:.1f}%', ha='center', 
               fontsize=10, fontweight='bold', color=color)
    
    ax.axvline(x=50, color=COLORS['danger'], linestyle='--', 
              linewidth=2, alpha=0.6, label='Random Baseline')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([45, 62])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig6_confidence_intervals.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 6: Confidence intervals (美化版)")

def main():
    print("\n" + "="*80)
    print("创建美化版论文图表")
    print("="*80 + "\n")
    
    os.makedirs('figures', exist_ok=True)
    
    fig1_performance_comparison()
    fig2_expert_usage()
    fig3_data_leakage()
    fig4_ablation_study()
    fig5_cross_market()
    fig6_confidence_intervals()
    
    print("\n" + "="*80)
    print("所有美化图表创建完成！")
    print("保存位置: ./figures/")
    print("="*80)
    print("\n配色方案:")
    print("  - 主色调: 深蓝 (#2E86AB)")
    print("  - 辅助色: 紫红 (#A23B72)")
    print("  - 强调色: 橙色 (#F18F01)")
    print("  - 成功: 绿色 (#06A77D)")
    print("  - 警告: 深橙 (#F77F00)")
    print("  - 危险: 红色 (#D62828)")
    print("\n特点:")
    print("  ✓ 专业的配色方案")
    print("  ✓ 清晰的数值标签")
    print("  ✓ 柔和的阴影和边框")
    print("  ✓ 高分辨率 (300 DPI)")
    print("  ✓ 适合论文发表")

if __name__ == '__main__':
    main()

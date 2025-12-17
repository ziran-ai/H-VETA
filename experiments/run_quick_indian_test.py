"""
Quick test on 3 representative Indian stocks
"""

import os
import sys

# Import the batch testing function
from run_indian_stocks_batch import test_single_stock
import json

def main():
    print("\n" + "="*80)
    print("Quick Test: 3 Representative Indian Stocks")
    print("="*80)
    
    # Select 3 diverse stocks
    stocks = [
        ('RELIANCE.csv', 'Energy/Conglomerate'),
        ('HDFCBANK.csv', 'Banking'),
        ('TCS.csv', 'IT Services')
    ]
    
    stock_dir = './data/indian_stocks_standard/'
    results = []
    
    for i, (stock_file, sector) in enumerate(stocks, 1):
        stock_path = os.path.join(stock_dir, stock_file)
        stock_name = stock_file.replace('.csv', '')
        
        print(f"\n[{i}/3] {stock_name} ({sector})")
        
        if not os.path.exists(stock_path):
            print(f"  ✗ File not found: {stock_path}")
            continue
        
        result = test_single_stock(stock_path, stock_name, num_epochs=30)  # Fewer epochs
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("QUICK TEST SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        import numpy as np
        accuracies = [r['test_directional_accuracy'] for r in successful]
        entropies = [r['entropy'] for r in successful]
        
        print(f"\nResults:")
        for r in successful:
            print(f"  {r['stock_name']}: {r['test_directional_accuracy']:.2%} (Entropy: {r['entropy']:.3f})")
        
        print(f"\nAverage Accuracy: {np.mean(accuracies):.2%}")
        print(f"Average Entropy: {np.mean(entropies):.3f}")
        print(f"All > 50%: {all(a > 0.50 for a in accuracies)}")
        print(f"No Collapse: {all(e > 0.5 for e in entropies)}")
    
    # Save
    with open('quick_indian_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: quick_indian_test_results.json")
    print("="*80)

if __name__ == '__main__':
    main()

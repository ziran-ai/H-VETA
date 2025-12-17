"""
Quick test on a single stock before running full batch
"""

from batch_experiment import train_single_stock
import json

def main():
    # Test on RELIANCE first (most representative Indian stock)
    print("\n" + "="*80)
    print("QUICK TEST: RELIANCE (India's largest company)")
    print("="*80)
    
    result = train_single_stock(
        stock_path='./data/indian_stocks_standard/RELIANCE.csv',
        stock_name='RELIANCE',
        num_epochs=50,
        save_dir='./checkpoints_test'
    )
    
    # Save result
    with open('test_reliance_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    if result.get('success'):
        print(f"\n✅ SUCCESS!")
        print(f"Directional Accuracy: {result['test_directional_accuracy']:.2%}")
        print(f"Expert Usage: {result['expert_usage_test']}")
        print(f"\nIf this looks good, run full batch with:")
        print(f"  python batch_experiment.py --num_epochs 50")
    else:
        print(f"\n❌ FAILED: {result.get('error')}")
        print(f"Need to fix issues before running full batch")

if __name__ == '__main__':
    main()

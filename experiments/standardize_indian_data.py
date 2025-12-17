"""
Standardize Indian stock data format to match GOOGL format
Indian data has: Date, Symbol, Series, Prev Close, Open, High, Low, Last, Close, VWAP, Volume, ...
Need: Date, Open, High, Low, Close, Adj Close, Volume
"""

import pandas as pd
import os
import glob

def standardize_indian_stock(input_path, output_path):
    """Convert Indian stock CSV to standard format"""
    try:
        df = pd.read_csv(input_path)
        
        # Check required columns
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            print(f"Missing columns in {input_path}")
            return False
        
        # Create standardized dataframe
        standard_df = pd.DataFrame({
            'Date': df['Date'],
            'Open': df['Open'],
            'High': df['High'],
            'Low': df['Low'],
            'Close': df['Close'],
            'Adj Close': df['Close'],  # Use Close as Adj Close
            'Volume': df['Volume']
        })
        
        # Remove rows with missing values
        standard_df = standard_df.dropna()
        
        # Sort by date
        standard_df['Date'] = pd.to_datetime(standard_df['Date'])
        standard_df = standard_df.sort_values('Date').reset_index(drop=True)
        
        # Save
        standard_df.to_csv(output_path, index=False)
        
        print(f"✓ Standardized {os.path.basename(input_path)}: {len(standard_df)} records")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("Standardizing Indian Stock Data")
    print("="*80)
    
    input_dir = './data/alldata/archive/'
    output_dir = './data/indian_stocks_standard/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    print(f"\nFound {len(csv_files)} stock files")
    
    success_count = 0
    for csv_file in csv_files:
        stock_name = os.path.basename(csv_file)
        output_path = os.path.join(output_dir, stock_name)
        
        if standardize_indian_stock(csv_file, output_path):
            success_count += 1
    
    print(f"\n" + "="*80)
    print(f"Standardization Complete: {success_count}/{len(csv_files)} successful")
    print(f"Output directory: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

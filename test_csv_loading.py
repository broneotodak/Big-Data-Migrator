"""
Test CSV file loading to debug why files aren't loading in the conversation system
"""
import pandas as pd
import os

def test_csv_loading():
    """Test loading CSV files directly"""
    
    print("🔍 Testing CSV File Loading")
    print("=" * 40)
    
    # File paths
    file1_path = os.path.abspath("temp/MMSDO_P_202412_EP810177.csv")
    file2_path = os.path.abspath("temp/Payments by order - 2024-12-01 - 2024-12-31.csv")
    
    print(f"File paths:")
    print(f"  📄 {file1_path}")
    print(f"  📄 {file2_path}")
    print(f"  Exists: {os.path.exists(file1_path)} | {os.path.exists(file2_path)}")
    
    # Test loading each file
    for i, file_path in enumerate([file1_path, file2_path], 1):
        print(f"\n{i}. Testing {os.path.basename(file_path)}:")
        
        try:
            # Check file properties
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   File size: {file_size} bytes")
                
                # Try reading the file
                df = pd.read_csv(file_path)
                print(f"   ✅ Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
                print(f"   Columns: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"   Columns: {list(df.columns)}")
                
                # Show first few rows
                print(f"   First row preview:")
                if len(df) > 0:
                    first_row = df.iloc[0]
                    for col, val in first_row.items():
                        print(f"     {col}: {val}")
                        if len(str(col)) > 50:  # Limit output
                            break
                
            else:
                print(f"   ❌ File does not exist")
                
        except Exception as e:
            print(f"   ❌ Error loading file: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            
            # Try with different parameters
            try:
                print(f"   🔧 Trying with encoding='latin1'...")
                df = pd.read_csv(file_path, encoding='latin1')
                print(f"   ✅ Success with latin1: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e2:
                print(f"   ❌ Still failed: {str(e2)}")
                
                try:
                    print(f"   🔧 Trying with different separator...")
                    df = pd.read_csv(file_path, sep=';')
                    print(f"   ✅ Success with semicolon: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e3:
                    print(f"   ❌ All attempts failed: {str(e3)}")

if __name__ == "__main__":
    test_csv_loading() 
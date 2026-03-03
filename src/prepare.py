import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_features, encode_labels, split_data

def prepare(input_path: str, output_dir: str):
    """
    Load raw data, preprocess, and split into train/test sets.
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    if 'label' in df.columns:
        X = df.drop('label', axis=1)
        y = df['label']
    else:
        raise ValueError("Input data must contain a 'label' column")

    # 1. Preprocess features
    print("Preprocessing features...")
    X_processed = preprocess_features(X)
    
    # 2. Encode labels
    print("Encoding labels...")
    y_encoded, label_encoder = encode_labels(y)
    
    # 3. Split data
    print("Splitting data...")
    # Use default test_size=0.2, random_state=42
    X_train, X_test, y_train, y_test = split_data(X_processed, y_encoded)
    
    # 4. Save prepared data
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and labels for saving
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    
    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Prepared data saved to {output_dir}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/prepare.py <input_file> <output_dir>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_directory = sys.argv[2]
    
    prepare(input_file, output_directory)

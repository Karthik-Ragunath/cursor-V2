#!/usr/bin/env python3
"""
Dataset exploration script for bespokelabs/bespoke-manim dataset
"""

import os
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter

def explore_dataset():
    """Explore the bespoke-manim dataset"""
    
    print("=" * 60)
    print("EXPLORING BESPOKE-MANIM DATASET")
    print("=" * 60)
    
    # Create cache directory
    cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Load dataset
        print("\n🔄 Loading dataset...")
        ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=cache_dir)
        print("✅ Dataset loaded successfully!")
        
        # Dataset overview
        print(f"\n📊 Dataset Structure:")
        print(f"Dataset type: {type(ds)}")
        print(f"Available splits: {list(ds.keys())}")
        
        # Explore each split
        for split_name, split_data in ds.items():
            print(f"\n" + "="*40)
            print(f"SPLIT: {split_name.upper()}")
            print(f"="*40)
            
            print(f"Number of examples: {len(split_data)}")
            print(f"Features: {split_data.features}")
            print(f"Column names: {split_data.column_names}")
            
            # Convert to pandas for easier exploration
            if len(split_data) > 0:
                # Sample a subset for exploration (first 1000 examples)
                sample_size = min(1000, len(split_data))
                sample_data = split_data.select(range(sample_size))
                df = sample_data.to_pandas()
                
                print(f"\n📈 Basic Statistics (sample of {sample_size} examples):")
                print(f"DataFrame shape: {df.shape}")
                
                # Show data types
                print(f"\n🔍 Data Types:")
                for col in df.columns:
                    print(f"  {col}: {df[col].dtype}")
                
                # # Show first few examples
                # print(f"\n📝 First 3 Examples:")
                # for i in range(min(3, len(df))):
                #     print(f"\nExample {i+1}:")
                #     for col in df.columns:
                #         value = df.iloc[i][col]
                #         if isinstance(value, str) and len(value) > 100:
                #             print(f"  {col}: {value[:100]}... (truncated)")
                #         else:
                #             print(f"  {col}: {value}")
                
                # Analyze text columns
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                if text_columns:
                    print(f"\n📊 Text Analysis:")
                    for col in text_columns:
                        if df[col].notna().any():
                            lengths = df[col].dropna().astype(str).str.len()
                            print(f"\n  {col}:")
                            print(f"    Non-null count: {df[col].notna().sum()}")
                            print(f"    Mean length: {lengths.mean():.1f}")
                            print(f"    Min length: {lengths.min()}")
                            print(f"    Max length: {lengths.max()}")
                            
                            # Show unique values if reasonable number
                            unique_count = df[col].nunique()
                            print(f"    Unique values: {unique_count}")
                            if unique_count <= 20:
                                print(f"    Values: {list(df[col].unique())}")
                
                # Look for missing values
                print(f"\n❌ Missing Values:")
                missing = df.isnull().sum()
                for col, count in missing.items():
                    if count > 0:
                        percentage = (count / len(df)) * 100
                        print(f"  {col}: {count} ({percentage:.1f}%)")
                
                # Sample random examples
                print(f"\n🎲 Random Sample (3 examples):")
                random_indices = np.random.choice(len(df), min(3, len(df)), replace=False)
                for i, idx in enumerate(random_indices):
                    print(f"\nRandom Example {i+1} (index {idx}):")
                    for col in df.columns:
                        value = df.iloc[idx][col]
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {col}: {value[:100]}... (truncated)")
                        else:
                            print(f"  {col}: {value}")
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed datasets: pip install datasets")
        print("2. Logged in to Hugging Face: huggingface-cli login")
        print("3. Have access to the dataset")
        
    print(f"\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)

def save_sample_data():
    """Save sample data to files for further analysis"""
    print("\n💾 Saving sample data...")
    
    try:
        cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
        ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=cache_dir)
        
        for split_name, split_data in ds.items():
            if len(split_data) > 0:
                # Save first 100 examples to CSV
                sample_size = min(100, len(split_data))
                sample_data = split_data.select(range(sample_size))
                df = sample_data.to_pandas()
                
                filename = f"sample_{split_name}.csv"
                df.to_csv(filename, index=False)
                print(f"✅ Saved {sample_size} examples from {split_name} to {filename}")
                
    except Exception as e:
        print(f"❌ Error saving sample data: {e}")

if __name__ == "__main__":
    # Explore the dataset
    explore_dataset()
    
    # Save sample data for further analysis
    # save_sample_data()
    
    print("\n🚀 Next steps:")
    print("1. Review the sample CSV files")
    print("2. Analyze the data structure and content")
    print("3. Design your training pipeline based on the data format")

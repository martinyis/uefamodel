"""
Step 1: Data Preprocessing for Champions League Match Prediction
Loads raw match data and creates a clean dataset with basic features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Create necessary directories
Path("data/processed").mkdir(parents=True, exist_ok=True)

def load_football_data(data_dir="data/raw"):
    """
    Load Champions League data from Football-Data.co.uk CSV files
    """
    all_matches = []
    
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        print("Please download data from https://www.football-data.co.uk/europem.php")
        print("Save the Champions League CSV files in the data/raw/ folder")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
            
            # Check if this looks like Champions League data
            required_cols = ['HomeTeam', 'AwayTeam']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {csv_file.name} - missing required columns")
                continue
            
            # Extract season from filename or use file date
            season = csv_file.stem
            df['Season'] = season
            
            all_matches.append(df)
            print(f"Loaded {len(df)} matches from {csv_file.name}")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if not all_matches:
        print("ERROR: No valid data files loaded!")
        return None
    
    # Combine all seasons
    df_combined = pd.concat(all_matches, ignore_index=True)
    print(f"\nTotal matches loaded: {len(df_combined)}")
    
    return df_combined

def clean_match_data(df):
    """
    Clean and standardize match data
    """
    print("\nCleaning data...")
    
    # Identify result column (could be FTR, Res, Result, etc.)
    result_col = None
    for col in ['FTR', 'Res', 'Result']:
        if col in df.columns:
            result_col = col
            break
    
    # Identify goals columns
    home_goals_col = None
    away_goals_col = None
    for col in ['FTHG', 'HG', 'HomeGoals']:
        if col in df.columns:
            home_goals_col = col
            break
    for col in ['FTAG', 'AG', 'AwayGoals']:
        if col in df.columns:
            away_goals_col = col
            break
    
    # Create standardized columns
    df_clean = pd.DataFrame()
    df_clean['Season'] = df['Season']
    df_clean['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df_clean['HomeTeam'] = df['HomeTeam'].str.strip()
    df_clean['AwayTeam'] = df['AwayTeam'].str.strip()
    
    if home_goals_col and away_goals_col:
        df_clean['HomeGoals'] = pd.to_numeric(df[home_goals_col], errors='coerce')
        df_clean['AwayGoals'] = pd.to_numeric(df[away_goals_col], errors='coerce')
    else:
        print("WARNING: Goals columns not found, creating dummy data")
        df_clean['HomeGoals'] = 0
        df_clean['AwayGoals'] = 0
    
    # Create result if not present
    if result_col:
        df_clean['Result'] = df[result_col]
    else:
        # Infer from goals
        df_clean['Result'] = 'D'  # Draw
        df_clean.loc[df_clean['HomeGoals'] > df_clean['AwayGoals'], 'Result'] = 'H'
        df_clean.loc[df_clean['HomeGoals'] < df_clean['AwayGoals'], 'Result'] = 'A'
    
    # Keep shots data if available
    for col in ['HS', 'AS', 'HST', 'AST']:
        if col in df.columns:
            df_clean[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'Result'])
    
    # Sort by date
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    print(f"Cleaned data: {len(df_clean)} valid matches")
    print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    print(f"\nResult distribution:")
    print(df_clean['Result'].value_counts())
    
    return df_clean

def add_basic_features(df):
    """
    Add basic match-level features
    """
    print("\nAdding basic features...")
    
    # Goal difference
    df['GoalDifference'] = df['HomeGoals'] - df['AwayGoals']
    
    # Total goals
    df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
    
    # Match ID for tracking
    df['MatchID'] = range(len(df))
    
    return df

def main():
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("CHAMPIONS LEAGUE DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    df_raw = load_football_data("data/raw")
    
    if df_raw is None:
        print("\n" + "="*60)
        print("SETUP REQUIRED:")
        print("="*60)
        print("1. Create a folder: data/raw/")
        print("2. Download Champions League CSV files from:")
        print("   https://www.football-data.co.uk/europem.php")
        print("3. Save them in data/raw/")
        print("4. Run this script again")
        return
    
    # Clean data
    df_clean = clean_match_data(df_raw)
    
    # Add basic features
    df_processed = add_basic_features(df_clean)
    
    # Save processed data
    output_file = "data/processed/matches_clean.csv"
    df_processed.to_csv(output_file, index=False)
    
    print(f"\n✓ Processed data saved to: {output_file}")
    print(f"✓ Shape: {df_processed.shape}")
    print("\nNext step: Run python 2_feature_engineering.py")

if __name__ == "__main__":
    main()



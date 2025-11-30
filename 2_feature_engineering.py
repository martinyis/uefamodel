"""
Step 2: Feature Engineering with Recency Weighting
Creates sophisticated features for predicting match outcomes
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_team_form(df, team, date, n_matches=5, recency_weight=True):
    """
    Calculate team's recent form with optional recency weighting
    
    Args:
        df: DataFrame with all matches
        team: Team name
        date: Current match date
        n_matches: Number of recent matches to consider
        recency_weight: If True, weight recent matches more heavily
    
    Returns:
        Dictionary with form statistics
    """
    # Get team's recent matches before this date
    home_matches = df[(df['HomeTeam'] == team) & (df['Date'] < date)].tail(n_matches)
    away_matches = df[(df['AwayTeam'] == team) & (df['Date'] < date)].tail(n_matches)
    
    # Combine and sort by date
    recent = pd.concat([home_matches, away_matches]).sort_values('Date').tail(n_matches)
    
    if len(recent) == 0:
        return {
            'goals_scored': 0,
            'goals_conceded': 0,
            'goal_diff': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'points': 0,
            'matches_played': 0
        }
    
    # Calculate weights if recency_weight is True
    if recency_weight and len(recent) > 0:
        # Exponential decay: most recent match has weight 1.0, older matches decay
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        weights = weights / weights.sum()  # Normalize to sum to 1
    else:
        weights = np.ones(len(recent)) / len(recent)
    
    # Calculate statistics
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    draws = 0
    losses = 0
    
    for idx, (i, match) in enumerate(recent.iterrows()):
        weight = weights[idx]
        
        if match['HomeTeam'] == team:
            goals_scored += match['HomeGoals'] * weight
            goals_conceded += match['AwayGoals'] * weight
            if match['Result'] == 'H':
                wins += weight
            elif match['Result'] == 'D':
                draws += weight
            else:
                losses += weight
        else:
            goals_scored += match['AwayGoals'] * weight
            goals_conceded += match['HomeGoals'] * weight
            if match['Result'] == 'A':
                wins += weight
            elif match['Result'] == 'D':
                draws += weight
            else:
                losses += weight
    
    points = (wins * 3) + draws
    
    return {
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'goal_diff': goals_scored - goals_conceded,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'points': points,
        'matches_played': len(recent)
    }

def calculate_head_to_head(df, home_team, away_team, date, n_matches=5):
    """
    Calculate head-to-head statistics between two teams
    """
    h2h = df[
        (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
         ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
        (df['Date'] < date)
    ].tail(n_matches)
    
    if len(h2h) == 0:
        return {
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_matches': 0
        }
    
    home_wins = len(h2h[(h2h['HomeTeam'] == home_team) & (h2h['Result'] == 'H')])
    home_wins += len(h2h[(h2h['AwayTeam'] == home_team) & (h2h['Result'] == 'A')])
    
    away_wins = len(h2h[(h2h['HomeTeam'] == away_team) & (h2h['Result'] == 'H')])
    away_wins += len(h2h[(h2h['AwayTeam'] == away_team) & (h2h['Result'] == 'A')])
    
    draws = len(h2h[h2h['Result'] == 'D'])
    
    return {
        'h2h_home_wins': home_wins,
        'h2h_draws': draws,
        'h2h_away_wins': away_wins,
        'h2h_matches': len(h2h)
    }

def engineer_features(df, n_form_matches=10, use_recency_weight=True):
    """
    Create comprehensive feature set for each match
    """
    print(f"\nEngineering features...")
    print(f"Using last {n_form_matches} matches for form calculation")
    print(f"Recency weighting: {'ON' if use_recency_weight else 'OFF'}")
    
    features_list = []
    
    for idx, match in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing match {idx+1}/{len(df)}", end='\r')
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        date = match['Date']
        
        # Home team form
        home_form = calculate_team_form(df, home_team, date, n_form_matches, use_recency_weight)
        
        # Away team form
        away_form = calculate_team_form(df, away_team, date, n_form_matches, use_recency_weight)
        
        # Head-to-head
        h2h = calculate_head_to_head(df, home_team, away_team, date)
        
        # Feature engineering
        features = {
            'MatchID': match['MatchID'],
            'Date': date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Result': match['Result'],
            
            # Home team features
            'home_goals_scored_avg': home_form['goals_scored'],
            'home_goals_conceded_avg': home_form['goals_conceded'],
            'home_goal_diff': home_form['goal_diff'],
            'home_points': home_form['points'],
            'home_wins': home_form['wins'],
            'home_draws': home_form['draws'],
            'home_matches_played': home_form['matches_played'],
            
            # Away team features
            'away_goals_scored_avg': away_form['goals_scored'],
            'away_goals_conceded_avg': away_form['goals_conceded'],
            'away_goal_diff': away_form['goal_diff'],
            'away_points': away_form['points'],
            'away_wins': away_form['wins'],
            'away_draws': away_form['draws'],
            'away_matches_played': away_form['matches_played'],
            
            # Differential features (key for prediction)
            'goal_diff_advantage': home_form['goal_diff'] - away_form['goal_diff'],
            'points_advantage': home_form['points'] - away_form['points'],
            'attack_vs_defense': home_form['goals_scored'] - away_form['goals_conceded'],
            'defense_vs_attack': away_form['goals_scored'] - home_form['goals_conceded'],
            
            # Head-to-head features
            'h2h_home_wins': h2h['h2h_home_wins'],
            'h2h_draws': h2h['h2h_draws'],
            'h2h_away_wins': h2h['h2h_away_wins'],
            'h2h_matches': h2h['h2h_matches'],
            
            # Team strength equality (for draw prediction)
            'strength_difference_abs': abs(home_form['goal_diff'] - away_form['goal_diff']),
            'points_difference_abs': abs(home_form['points'] - away_form['points']),
        }
        
        features_list.append(features)
    
    print(f"\nProcessed all {len(df)} matches")
    
    df_features = pd.DataFrame(features_list)
    
    # Add interaction terms
    df_features['form_interaction'] = (
        df_features['home_goal_diff'] * df_features['away_goal_diff']
    )
    
    return df_features

def main():
    """
    Main feature engineering pipeline
    """
    print("="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Load preprocessed data
    input_file = "data/processed/matches_clean.csv"
    
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded {len(df)} matches from {input_file}")
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found!")
        print("Run: python 1_data_preprocessing.py first")
        return
    
    # Create features with recency weighting (as per methodology)
    df_features = engineer_features(df, n_form_matches=10, use_recency_weight=True)
    
    # Save features
    output_file = "data/processed/features.csv"
    df_features.to_csv(output_file, index=False)
    
    print(f"\n✓ Features saved to: {output_file}")
    print(f"✓ Shape: {df_features.shape}")
    print(f"✓ Number of features: {df_features.shape[1] - 4}")  # Exclude ID, Date, Teams, Result
    
    # Summary statistics
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(f"Total matches: {len(df_features)}")
    print(f"Matches with sufficient history: {len(df_features[df_features['home_matches_played'] >= 5])}")
    print(f"\nResult distribution:")
    print(df_features['Result'].value_counts())
    
    print("\nNext step: Run python 3_train_model.py")

if __name__ == "__main__":
    main()



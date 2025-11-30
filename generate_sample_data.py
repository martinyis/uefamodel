"""
Generate realistic sample Champions League data for demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Champions League teams
teams = [
    'Bayern Munich', 'Real Madrid', 'Barcelona', 'Man City', 'Liverpool',
    'Chelsea', 'PSG', 'Juventus', 'Atletico Madrid', 'Dortmund',
    'Inter Milan', 'AC Milan', 'Tottenham', 'Man United', 'Ajax',
    'Porto', 'Benfica', 'RB Leipzig', 'Sevilla', 'Lyon'
]

# Team strengths (0-1, higher = stronger)
team_strength = {
    'Bayern Munich': 0.9, 'Real Madrid': 0.88, 'Barcelona': 0.85, 'Man City': 0.92,
    'Liverpool': 0.87, 'Chelsea': 0.82, 'PSG': 0.86, 'Juventus': 0.80,
    'Atletico Madrid': 0.78, 'Dortmund': 0.75, 'Inter Milan': 0.76, 'AC Milan': 0.74,
    'Tottenham': 0.73, 'Man United': 0.77, 'Ajax': 0.72, 'Porto': 0.68,
    'Benfica': 0.67, 'RB Leipzig': 0.74, 'Sevilla': 0.71, 'Lyon': 0.69
}

def simulate_match(home_team, away_team, date):
    """Simulate a realistic match result based on team strengths"""
    home_strength = team_strength[home_team]
    away_strength = team_strength[away_team]
    
    # Home advantage
    home_strength += 0.1
    
    # Expected goals based on team strength
    home_xg = home_strength * 3 + np.random.normal(0, 0.5)
    away_xg = away_strength * 3 + np.random.normal(0, 0.5)
    
    # Generate actual goals (Poisson-like but simplified)
    home_goals = max(0, int(np.random.poisson(max(0, home_xg))))
    away_goals = max(0, int(np.random.poisson(max(0, away_xg))))
    
    # Determine result
    if home_goals > away_goals:
        result = 'H'
    elif away_goals > home_goals:
        result = 'A'
    else:
        result = 'D'
    
    # Generate shots (correlated with goals)
    home_shots = home_goals * 3 + np.random.randint(3, 10)
    away_shots = away_goals * 3 + np.random.randint(3, 10)
    home_shots_target = int(home_shots * 0.4) + home_goals
    away_shots_target = int(away_shots * 0.4) + away_goals
    
    return {
        'Date': date.strftime('%d/%m/%Y'),
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'FTHG': home_goals,
        'FTAG': away_goals,
        'FTR': result,
        'HS': home_shots,
        'AS': away_shots,
        'HST': home_shots_target,
        'AST': away_shots_target
    }

def generate_season(start_date, season_name):
    """Generate a season of Champions League matches"""
    matches = []
    current_date = start_date
    
    # Generate ~125 matches per season (realistic for Champions League)
    for _ in range(125):
        # Pick two different teams
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        match = simulate_match(home_team, away_team, current_date)
        matches.append(match)
        
        # Advance date (matches are typically weekly)
        current_date += timedelta(days=np.random.randint(3, 10))
    
    df = pd.DataFrame(matches)
    filename = f'data/raw/champions_league_{season_name}.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Generated {len(df)} matches for {season_name} season")
    return df

# Generate multiple seasons
print("Generating sample Champions League data...")
print("="*60)

seasons = [
    (datetime(2019, 9, 1), '2019_2020'),
    (datetime(2020, 9, 1), '2020_2021'),
    (datetime(2021, 9, 1), '2021_2022'),
    (datetime(2022, 9, 1), '2022_2023'),
]

all_matches = []
for start_date, season_name in seasons:
    df = generate_season(start_date, season_name)
    all_matches.append(df)

total_matches = sum(len(df) for df in all_matches)
print("="*60)
print(f"✓ Total matches generated: {total_matches}")
print(f"✓ Files saved in data/raw/")
print("\nYou can now run: python RUN_ALL.py")


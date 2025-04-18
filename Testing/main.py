import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate

import xgboost as xgb

# =============================================================================
# 1. Data Collection and Data Cleaning
# =============================================================================

# Import preprocessing functions
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load preprocessed NCAA datasets
data_dir = os.path.join(project_root, 'MarchMadnessData')
team_spellings = pd.read_csv(f'{data_dir}/MTeamSpellings.csv')
tourney_seeds = pd.read_csv(f'{data_dir}/MNCAATourneySeeds.csv')
team_conferences = pd.read_csv(f'{data_dir}/MTeamConferences.csv')
reg_results = pd.read_csv(f'{data_dir}/MRegularSeasonDetailedResults.csv')
tourney_results = pd.read_csv(f'{data_dir}/MNCAATourneyCompactResults.csv')
seed_results = pd.read_csv(f'{data_dir}/SeedPerformanceReport.csv')
seed_results = seed_results.set_index("SEED")

# ----------------------------
# Process Regular Season Data
# ----------------------------
# Compute per-team averages for each season using winner and loser info.
winners = reg_results[['Season', 'WTeamID', 'WScore', 'LScore']].copy()
winners.rename(columns={'WTeamID':'TeamID', 'WScore':'Points', 'LScore':'OppPoints'}, inplace=True)
losers = reg_results[['Season', 'LTeamID', 'LScore', 'WScore']].copy()
losers.rename(columns={'LTeamID':'TeamID', 'LScore':'Points', 'WScore':'OppPoints'}, inplace=True)
all_games = pd.concat([winners, losers], ignore_index=True)
team_stats = all_games.groupby(['Season', 'TeamID'], as_index=False).agg({'Points': 'mean', 'OppPoints': 'mean'})
team_stats.rename(columns={'Points':'AvgPoints', 'OppPoints':'AvgOppPoints'}, inplace=True)
team_stats['AvgMargin'] = team_stats['AvgPoints'] - team_stats['AvgOppPoints']

# ----------------------------
# Process Tournament Seeds
# ----------------------------
# Extract numeric seed values (e.g., "W01" becomes 1)
def extract_seed(seed_str):
    match = re.search(r'[W-Z](\d+)$', seed_str)
    return int(match.group(1)) if match else np.nan

tourney_seeds['SeedNum'] = tourney_seeds['Seed'].apply(extract_seed)
tourney_seeds = tourney_seeds.dropna(subset=['SeedNum'])
# After dropping NaN values, safely convert to integers
tourney_seeds['SeedNum'] = tourney_seeds['SeedNum'].astype(int)


# tourney_seeds = tourney_seeds[tourney_seeds['SeedNum'].notna()]

# =============================================================================
# 2. Build the Matchup Training Dataset from Tournament Results
# =============================================================================
# We'll create two training examples per game: one from the winner's perspective (target=1)
# and one from the loser's perspective (target=0).

# Merge seed info for both teams from tourney_seeds (keeping Season and TeamID)
seeds = tourney_seeds[['Season', 'TeamID', 'SeedNum']].copy()
team_features = pd.merge(team_stats, seeds, on=['Season', 'TeamID'], how='left')

def get_team_features(season, team_id):
    feat = team_features[(team_features['Season'] == season) & (team_features['TeamID'] == team_id)]
    feat = feat.dropna(subset=["SeedNum"])
    feat['SeedNum'] = feat['SeedNum'].astype(int)
    return feat.iloc[0] if not feat.empty else None

training_rows = []
for idx, row in tourney_results.iterrows():
    season = row['Season']
    winner_id = row['WTeamID']
    loser_id = row['LTeamID']
    
    winner_feat = get_team_features(season, winner_id)
    loser_feat = get_team_features(season, loser_id)
    
    if winner_feat is not None and loser_feat is not None:
        # Create rows from both perspectives
        training_rows.append({
            'Team1ID': winner_id,
            'Team2ID': loser_id,
            'Team1_AvgPoints': winner_feat['AvgPoints'],
            'Team1_AvgOppPoints': winner_feat['AvgOppPoints'],
            'Team1_SeedNum': winner_feat['SeedNum'],
            'Team1_SeedWin%': seed_results.loc[winner_feat['SeedNum'], "WIN%"],
            'Team1_SeedR1%': seed_results.loc[winner_feat['SeedNum'], "R64%"],
            'Team1_SeedR2%': seed_results.loc[winner_feat['SeedNum'], "R32%"],
            'Team2_AvgPoints': loser_feat['AvgPoints'],
            'Team2_AvgOppPoints': loser_feat['AvgOppPoints'],
            'Team2_SeedNum': loser_feat['SeedNum'],
            'Team2_SeedWin%': seed_results.loc[loser_feat['SeedNum'], "WIN%"],
            'Team2_SeedR1%': seed_results.loc[loser_feat['SeedNum'], "R64%"],
            'Team2_SeedR2%': seed_results.loc[loser_feat['SeedNum'], "R32%"],
            'Target': 1
        })
        
        training_rows.append({
            'Team1ID': loser_id,
            'Team2ID': winner_id,
            'Team1_AvgPoints': loser_feat['AvgPoints'],
            'Team1_AvgOppPoints': loser_feat['AvgOppPoints'],
            'Team1_SeedNum': loser_feat['SeedNum'],
            'Team1_SeedWin%': seed_results.loc[loser_feat['SeedNum'].astype(int), "WIN%"],
            'Team1_SeedR1%': seed_results.loc[loser_feat['SeedNum'].astype(int), "R64%"],
            'Team1_SeedR2%': seed_results.loc[loser_feat['SeedNum'].astype(int), "R32%"],
            'Team2_AvgPoints': winner_feat['AvgPoints'],
            'Team2_AvgOppPoints': winner_feat['AvgOppPoints'],
            'Team2_SeedNum': winner_feat['SeedNum'],
            'Team2_SeedWin%': seed_results.loc[winner_feat['SeedNum'].astype(int), "WIN%"],
            'Team2_SeedR1%': seed_results.loc[winner_feat['SeedNum'].astype(int), "R64%"],
            'Team2_SeedR2%': seed_results.loc[winner_feat['SeedNum'].astype(int), "R32%"],
            'Target': 0
        })

training_df = pd.DataFrame(training_rows)
print("Training dataset shape:", training_df.shape)
print(training_df.head())

# =============================================================================
# 3. Model Training & Evaluation
# =============================================================================

# Split features and target
feature_cols = ['Team1_AvgPoints', 'Team1_AvgOppPoints', 'Team1_SeedNum',
                'Team1_SeedWin%', 'Team1_SeedR1%', 'Team1_SeedR2%', 
                'Team2_AvgPoints', 'Team2_AvgOppPoints', 'Team2_SeedNum', 
                'Team2_SeedWin%', 'Team2_SeedR1%', 'Team2_SeedR2%']
features = training_df[feature_cols]
labels = training_df['Target']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Define model candidates for each round
# Create early round models
early_models = {
    'logistic': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    'xgboost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
}

# Create middle round models - slightly different parameters than early round models
middle_models = {
    'logistic': LogisticRegression(C=10.0, max_iter=1500, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=250, max_depth=25, random_state=42),
    'xgboost': xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.9, random_state=42)
}

# Create final round models - optimized parameters for championship prediction
final_models = {
    'logistic': LogisticRegression(C=50.0, max_iter=2000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=400, max_depth=30, min_samples_split=4, random_state=42),
    'xgboost': xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, random_state=42)
}

# Function to evaluate models
def evaluate_models(models, X_train, X_test, y_train, y_test, round_name):
    print(f"\nEvaluating {round_name} models...")
    results = {}
    best_score = 0
    best_model = None
    feature_importances = {}
    
    for name, model in models.items():
        start_time = time.time()
        # Train and evaluate the model
        model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Update best model if this one is better
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
        
        # Calculate feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
            
        training_time = time.time() - start_time
        print(f"\n{name.upper()} Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Show feature importances for tree-based models
    for name, importances in feature_importances.items():
        print(f"\nFeature importances for {name}:")
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_names))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Create an enhanced visualization of model performance
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Get model names and accuracies
    model_names = list(results.keys())
    accuracies = [results[name] for name in model_names]
    model_names = [name.upper() for name in model_names]  # Uppercase for display
    
    # Create enhanced bar plot
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(accuracies)))
    bars = plt.bar(model_names, accuracies, color=colors, width=0.6, edgecolor='gray', linewidth=1)
    
    # Add grid, title and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'Model Accuracies for {round_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Models', fontsize=12, labelpad=10)
    plt.ylabel('Accuracy', fontsize=12, labelpad=10)
    
    # Add text for exact values on top of bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add a title showing best model
    plt.annotate(f'Best Model: {model_names[accuracies.index(max(accuracies))]} ({max(accuracies):.4f})',
                xy=(0.5, 0.95), xycoords='axes fraction', fontsize=12,
                ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Set y-axis to start at an appropriate value
    plt.ylim(min(accuracies) * 0.95, max(accuracies) * 1.05)
    
    # Rotate x-axis labels slightly for better readability if needed
    plt.xticks(rotation=0, fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10)
    
    # Add a subtle outer frame
    plt.box(True)
    
    # Save the bar chart
    plt.tight_layout()
    plt.savefig(f'Modeling/{round_name.lower().replace(" ", "_")}_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBest {round_name} model: {type(best_model).__name__}")
    print(f"Best accuracy: {best_score:.4f}")
    return best_model

early_model = evaluate_models(early_models, X_train_scaled, X_test_scaled, 
                            y_train, y_test, "Early Round")

for index, row in X_train.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "S16%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "E8%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "S16%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "E8%"]

for index, row in X_test.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "S16%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "E8%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "S16%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "E8%"]

# Evaluate middle round models (using all features)
middle_model = evaluate_models(middle_models, X_train_scaled, X_test_scaled, 
                              y_train, y_test, "Middle Round")

for _, row in X_train.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F4%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F2%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F4%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F2%"]

for _, row in X_test.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F4%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F2%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F4%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F2%"]


# Evaluate final round models (using all features)
final_model = evaluate_models(final_models, X_train_scaled, X_test_scaled, 
                             y_train, y_test, "Final Round")

# =============================================================================
# 4. Simulation and Bracket Prediction
# =============================================================================

class Bracket:
    def __init__(self, season, model, scaler_obj=None, current_round=1):
        self.season = season
        self.model = model
        self.scaler = scaler_obj
        self.rounds = {}
        self.current_round = current_round
        self.team_names = pd.read_csv(os.path.join(project_root, 'MarchMadnessData', 'MTeams.csv'))
        self.seed_data = pd.read_csv(os.path.join(project_root, 'MarchMadnessData', 'SeedPerformanceReport.csv'))
        self.seed_data = self.seed_data.set_index("SEED")
        
    
    def simulate_matchup(self, team1_id, team2_id):
        team1_feat = get_team_features(self.season, team1_id)
        team2_feat = get_team_features(self.season, team2_id)
        
        if team1_feat is None or team2_feat is None:
            print(f"Missing features for team {team1_id} or {team2_id} in season {self.season}")
            # Return default values when features are missing
            return team1_id, 0.5, 0.5  # Default to 50/50 probability when features are missing
            
        # Create feature array with all stats
        features = pd.DataFrame([
            {
                'Team1_AvgPoints': team1_feat['AvgPoints'],
                'Team1_AvgOppPoints': team1_feat['AvgOppPoints'],
                'Team1_SeedNum': team1_feat['SeedNum'],
                'Team1_SeedWin%': self.seed_data.at[team1_feat['SeedNum'].astype(int), 'WIN%'],
                'Team1_SeedR1%': self.seed_data.at[team1_feat['SeedNum'].astype(int), 'R64%'],
                'Team1_SeedR2%': self.seed_data.at[team1_feat['SeedNum'].astype(int), 'R32%'],
                'Team2_AvgPoints': team2_feat['AvgPoints'],
                'Team2_AvgOppPoints': team2_feat['AvgOppPoints'],
                'Team2_SeedNum': team2_feat['SeedNum'],
                'Team2_SeedWin%': self.seed_data.at[team2_feat['SeedNum'].astype(int), 'WIN%'],
                'Team2_SeedR1%': self.seed_data.at[team2_feat['SeedNum'].astype(int), 'R64%'],
                'Team2_SeedR2%': self.seed_data.at[team2_feat['SeedNum'].astype(int), 'R32%']
            }
        ])
        
        # Scale features if scaler is provided
        if self.scaler is not None:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns
            )
        
        # Get probability predictions based on current round
        probs = self.model.predict_proba(features, self.current_round)
        
        # Extract probabilities based on tournament round
        if self.current_round <= 2:
            model_probs = probs['early']
        elif self.current_round <= 4:
            model_probs = probs['middle']
        else:
            model_probs = probs['final']
        
        # Return winner and both teams' probabilities
        team1_prob = model_probs[0][1]
        team2_prob = model_probs[0][0]
        winner_id = team1_id if team1_prob > team2_prob else team2_id
        return winner_id, team1_prob, team2_prob
    
    def get_team_name(self, team_id):
        team = self.team_names[self.team_names['TeamID'] == team_id]
        return team['TeamName'].iloc[0] if not team.empty else f"Team {team_id}"

class BracketSimulation:
    def __init__(self, season, model, scaler_obj=None, num_simulations=1000):
        self.season = season
        self.model = model
        self.scaler = scaler_obj
        self.num_simulations = num_simulations
        self.brackets = []
        self.matchup_stats = {}
        self.current_round = 1
    
    def simulate_brackets(self):
        # Get tournament teams for the season
        tourney_slots = pd.read_csv(os.path.join(project_root, 'MarchMadnessData', 'MNCAATourneySlots.csv'))
        season_seeds = tourney_slots[tourney_slots['Season'] == self.season]
        
        # Initialize regions and bracket state
        self.regions = {'W': [], 'X': [], 'Y': [], 'Z': []}
        self.region_winners = {}
        self.championship_matchup = None
        self.champion = None
        self.bracket_results = {
            1: [],  # First Round results (64 teams, 32 matchups)
            2: [],  # Second Round results (32 teams, 16 matchups)
            3: [],  # Sweet 16 results (16 teams, 8 matchups)
            4: [],  # Elite 8 results (8 teams, 4 matchups)
            5: [],  # Final Four results (4 teams, 2 matchups)
            6: []   # Championship result (2 teams, 1 matchup)
        }
        
        # Group teams by region
        for _, row in season_seeds.iterrows():
            strong_region_code = row['StrongSeed'][0]  # Extract W, X, Y, Z
            # Handle non-standard seed formats like '16a' by using regex to extract just the numeric part
            import re
            strong_seed_match = re.search(r'^\d+$', row['StrongSeed'][1:])
            
            if strong_seed_match:
                strong_seed_num = int(strong_seed_match.group())
            else:
                strong_seed_num = False
            strong_team_id = row['StrongID']
            # Store as (team_id, seed_num) tuples in each region
            if strong_region_code in self.regions and strong_seed_num:
                self.regions[strong_region_code].append((strong_team_id, strong_seed_num))

            weak_region_code = row['WeakSeed'][0]  # Extract W, X, Y, Z
            weak_seed_match = re.search(r'^\d+$', row['WeakSeed'][1:])

            if weak_seed_match:
                weak_seed_num = int(weak_seed_match.group())
            else:
                weak_seed_num = False
            weak_team_id = row['WeakID']
            if weak_region_code in self.regions and weak_seed_num:
                self.regions[weak_region_code].append((weak_team_id, weak_seed_num))
        
        # Sort teams by seed number within each region
        for region in self.regions:
            self.regions[region] = sorted(self.regions[region], key=lambda x: x[1])
            # Print count of teams in each region
            print(f"{region} region: {len(self.regions[region])} teams")
        
        # Simulate regional brackets (Rounds 1-4)
        self.simulate_regional_brackets()
        
        # Simulate Final Four (Round 5)
        self.simulate_final_four()
        
        # Simulate Championship (Round 6)
        self.simulate_championship()
    
    def simulate_regional_brackets(self):
        # Simulate rounds 1-4 within each region
        for region_code, teams in self.regions.items():
            print(f"Simulating {region_code} region...")
            
            # Ensure we have all 16 seeds for this region
            if len(teams) < 16:
                print(f"Warning: {region_code} region has fewer than 16 teams ({len(teams)} teams)")
                # If needed, could add placeholder teams here
            
            # Sort teams by seed to ensure proper ordering
            teams.sort(key=lambda x: x[1])  # Sort by seed number
            
            # Create play-in games for 16, 11 seeds if necessary (not implemented here)
            # For simplicity, we'll use the regular 1-16 seeding matchups
            
            # Round 1: Create all first round matchups (1 vs 16, 8 vs 9, etc.)
            round1_matchups = []
            # Standard NCAA tournament first round matchups:
            matchup_pairs = [
                (1, 16), (8, 9),    # Top quarter of bracket
                (5, 12), (4, 13),   # Second quarter of bracket
                (6, 11), (3, 14),   # Third quarter of bracket
                (7, 10), (2, 15)    # Bottom quarter of bracket
            ]
            
            # Create matchups based on seed numbers
            for seed1, seed2 in matchup_pairs:
                # Find teams with these seeds
                team1 = next((t for t in teams if t[1] == seed1), None)
                # print(f"Team1: {team1}")
                team2 = next((t for t in teams if t[1] == seed2), None)
                # print(f"Team2: {team2}")
                
                if team1 and team2:  # Ensure both teams exist
                    round1_matchups.append(((team1[0], team1[1]), (team2[0], team2[1])))
                else:
                    print(f"Warning: Could not find teams for matchup {seed1} vs {seed2} in {region_code} region")
                    # In a real implementation, we might handle this better
            
            # Store all matchups for this region with region code
            region_results = {
                1: [],  # First round results
                2: [],  # Second round results
                3: [],  # Sweet 16 results
                4: []   # Elite 8 results
            }
            
            # Simulate Round 1
            self.current_round = 1
            round1_winners = []
            for matchup in round1_matchups:
                team1_id, team1_seed = matchup[0]
                team2_id, team2_seed = matchup[1]
                
                # Simulate this matchup once and store probabilities
                bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
                winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
                self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
                winner_id = team1_id if team1_prob > team2_prob else team2_id
                winner_seed = team1_seed if team1_prob > team2_prob else team2_seed
                round1_winners.append((winner_id, winner_seed))
                region_results[1].append(((team1_id, team1_seed), (team2_id, team2_seed), winner_id))
                self.bracket_results[1].append(((region_code, team1_id, team1_seed), (region_code, team2_id, team2_seed), (region_code, winner_id)))
            
            # Add one bracket instance
            if not self.brackets:
                self.brackets.append(bracket)
            
            # Simulate Round 2 (Second Round)
            self.current_round = 2

            # Sorting lower seed as team 1 and higher seed as team 2
            r2_match1 = (round1_winners[0], round1_winners[1]) if round1_winners[0][1] <= round1_winners[1][1] else (round1_winners[1], round1_winners[0])
            r2_match2 = (round1_winners[2], round1_winners[3]) if round1_winners[2][1] <= round1_winners[3][1] else (round1_winners[3], round1_winners[2])
            r2_match3 = (round1_winners[4], round1_winners[5]) if round1_winners[4][1] <= round1_winners[5][1] else (round1_winners[5], round1_winners[4])
            r2_match4 = (round1_winners[6], round1_winners[7]) if round1_winners[6][1] <= round1_winners[7][1] else (round1_winners[7], round1_winners[6])

            round2_matchups = [
                r2_match1, # Winners of 1v16 and 8v9
                r2_match2,  # Winners of 5v12 and 4v13
                r2_match3,  # Winners of 6v11 and 3v14
                r2_match4   # Winners of 7v10 and 2v15
            ]
            
            round2_winners = []
            for matchup in round2_matchups:
                team1_id, team1_seed = matchup[0]
                team2_id, team2_seed = matchup[1]

                if (team1_id, team2_id) not in self.matchup_stats and (team2_id, team1_id) not in self.matchup_stats:
                    self.matchup_stats[(team1_id, team2_id)] = {team1_id: 0, team2_id: 0}
                else:
                    print(f"Round {self.current_round} Matchup already simulated: {team1_id} vs {team2_id}")
                
                # Simulate this matchup once and store probabilities
                bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
                winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
                self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
                winner_id = team1_id if team1_prob > team2_prob else team2_id
                winner_seed = team1_seed if team1_prob > team2_prob else team2_seed
                round2_winners.append((winner_id, winner_seed))
                region_results[2].append(((team1_id, team1_seed), (team2_id, team2_seed), winner_id))
                self.bracket_results[2].append(((region_code, team1_id, team1_seed), (region_code, team2_id, team2_seed), (region_code, winner_id)))
            
            # Simulate Round 3 (Sweet 16)
            self.current_round = 3

            # Sorting lower seed as team 1 and higher seed as team 2
            r3_match1 = (round2_winners[0], round2_winners[1]) if round2_winners[0][1] <= round2_winners[1][1] else (round2_winners[1], round2_winners[0])
            r3_match2 = (round2_winners[2], round2_winners[3]) if round2_winners[2][1] <= round2_winners[3][1] else (round2_winners[3], round2_winners[2])
            round3_matchups = [
                r3_match1,  # Winners from top half of bracket
                r3_match2   # Winners from bottom half of bracket
            ]
            
            round3_winners = []
            for matchup in round3_matchups:
                team1_id, team1_seed = matchup[0]
                team2_id, team2_seed = matchup[1]

                if (team1_id, team2_id) not in self.matchup_stats and (team2_id, team1_id) not in self.matchup_stats:
                    self.matchup_stats[(team1_id, team2_id)] = {team1_id: 0, team2_id: 0}
                else:
                    print(f"Round {self.current_round} Matchup already simulated: {team1_id} vs {team2_id}")
                
                # Simulate this matchup once and store probabilities
                bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
                winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
                self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
                winner_id = team1_id if team1_prob > team2_prob else team2_id
                winner_seed = team1_seed if team1_prob > team2_prob else team2_seed
                round3_winners.append((winner_id, winner_seed))
                region_results[3].append(((team1_id, team1_seed), (team2_id, team2_seed), winner_id))
                self.bracket_results[3].append(((region_code, team1_id, team1_seed), (region_code, team2_id, team2_seed), (region_code, winner_id)))
            
            # Simulate Round 4 (Elite 8 - Regional Final)
            self.current_round = 4
            r4_match = (round3_winners[0], round3_winners[1]) if round3_winners[0][1] <= round3_winners[1][1] else (round3_winners[1], round3_winners[0])
            team1_id, team1_seed = r4_match[0]
            team2_id, team2_seed = r4_match[1]

            if (team1_id, team2_id) not in self.matchup_stats and (team2_id, team1_id) not in self.matchup_stats:
                self.matchup_stats[(team1_id, team2_id)] = {team1_id: 0, team2_id: 0}
            else:
                print(f"Round {self.current_round} Matchup already simulated: {team1_id} vs {team2_id}")

            # Simulate this matchup once and store probabilities
            bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
            winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
            self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
            winner_id = team1_id if team1_prob > team2_prob else team2_id
            winner_seed = team1_seed if team1_prob > team2_prob else team2_seed
            region_results[4].append(((team1_id, team1_seed), (team2_id, team2_seed), winner_id))
            self.bracket_results[4].append(((region_code, team1_id, team1_seed), (region_code, team2_id, team2_seed), (region_code, winner_id)))
            self.region_winners[region_code] = (winner_id, winner_seed)
            
            print(f"{region_code} Region Winner: {self.brackets[0].get_team_name(winner_id)} (Seed: {winner_seed})")
        
    def simulate_final_four(self):
        # Simulate Round 5 (Final Four)
        print("\nSimulating Final Four...")
        self.current_round = 5
        
        # Traditional Final Four matchups: W vs X, Y vs Z
        # These pairings can be adjusted based on actual tournament structure
        semifinal1 = ((self.region_winners['W'][0], self.region_winners['W'][1], 'W'), 
                     (self.region_winners['X'][0], self.region_winners['X'][1], 'X'))
        semifinal2 = ((self.region_winners['Y'][0], self.region_winners['Y'][1], 'Y'), 
                     (self.region_winners['Z'][0], self.region_winners['Z'][1], 'Z'))
        
        # Simulate first semifinal
        r5_match1 = (semifinal1[0], semifinal1[1]) if semifinal1[0][1] <= semifinal1[1][1] else (semifinal1[1], semifinal1[0])
        team1_id, team1_seed, team1_region = r5_match1[0]
        team2_id, team2_seed, team2_region = r5_match1[1]

        if (team1_id, team2_id) not in self.matchup_stats and (team2_id, team1_id) not in self.matchup_stats:
            self.matchup_stats[(team1_id, team2_id)] = {team1_id: 0, team2_id: 0}
        else:
            print(f"Round {self.current_round} Matchup already simulated: {team1_id} vs {team2_id}")
        
        # Simulate this matchup once and store probabilities
        bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
        winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
        self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
        finalist1_id = team1_id if team1_prob > team2_prob else team2_id
        finalist1_seed = team1_seed if team1_prob > team2_prob else team2_seed
        finalist1_region = team1_region if team1_prob > team2_prob else team2_region
        
        self.bracket_results[5].append(((team1_region, team1_id, team1_seed), (team2_region, team2_id, team2_seed), (finalist1_region, finalist1_id)))
        
        # Simulate second semifinal
        r5_match2 = (semifinal2[0], semifinal2[1]) if semifinal2[0][1] <= semifinal2[1][1] else (semifinal2[1], semifinal2[0])
        team1_id, team1_seed, team1_region = r5_match2[0]
        team2_id, team2_seed, team2_region = r5_match2[1]
        
        if (team1_id, team2_id) not in self.matchup_stats and (team2_id, team1_id) not in self.matchup_stats:
            self.matchup_stats[(team1_id, team2_id)] = {team1_id: 0, team2_id: 0}
        else:
            print(f"Round {self.current_round} Matchup already simulated: {team1_id} vs {team2_id}")
        
        # Simulate this matchup once and store probabilities
        bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
        winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
        self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
        finalist2_id = team1_id if team1_prob > team2_prob else team2_id
        finalist2_seed = team1_seed if team1_prob > team2_prob else team2_seed
        finalist2_region = team1_region if team1_prob > team2_prob else team2_region
        
        self.bracket_results[5].append(((team1_region, team1_id, team1_seed), (team2_region, team2_id, team2_seed), (finalist2_region, finalist2_id)))
        
        # Store championship matchup
        self.championship_matchup = ((finalist1_id, finalist1_seed, finalist1_region), 
                                    (finalist2_id, finalist2_seed, finalist2_region))
        
        print(f"First Finalist: {self.brackets[0].get_team_name(finalist1_id)} ({finalist1_region} Region, Seed: {finalist1_seed})")
        print(f"Second Finalist: {self.brackets[0].get_team_name(finalist2_id)} ({finalist2_region} Region, Seed: {finalist2_seed})")
    
    def simulate_championship(self):
        # Simulate Round 6 (Championship)
        print("\nSimulating Championship...")
        self.current_round = 6
        
        # Get championship matchup
        r6_matchup = (self.championship_matchup[0], self.championship_matchup[1]) if self.championship_matchup[0][1] <= self.championship_matchup[1][1] else (self.championship_matchup[1], self.championship_matchup[0])
        team1_id, team1_seed, team1_region = r6_matchup[0]
        team2_id, team2_seed, team2_region = r6_matchup[1]
        
        if (team1_id, team2_id) not in self.matchup_stats and (team2_id, team1_id) not in self.matchup_stats:
            self.matchup_stats[(team1_id, team2_id)] = {team1_id: 0, team2_id: 0}
        else: 
            print(f"Round {self.current_round} Matchup already simulated: {team1_id} vs {team2_id}")
        
        # Simulate this matchup once and store probabilities
        bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
        winner, team1_prob, team2_prob = bracket.simulate_matchup(team1_id, team2_id)
        self.matchup_stats[(team1_id, team2_id)] = {team1_id: team1_prob, team2_id: team2_prob}
        champion_id = team1_id if team1_prob > team2_prob else team2_id
        champion_seed = team1_seed if team1_prob > team2_prob else team2_seed
        champion_region = team1_region if team1_prob > team2_prob else team2_region
        
        self.bracket_results[6].append(((team1_region, team1_id, team1_seed), (team2_region, team2_id, team2_seed), (champion_region, champion_id)))
        
        # Store the champion
        self.champion = (champion_id, champion_seed, champion_region)
        
        print(f"Champion: {self.brackets[0].get_team_name(champion_id)} ({champion_region} Region, Seed: {champion_seed})")
    
    def visualize_matchup_stats(self):
        # Get a sample bracket to use for team name lookups
        sample_bracket = self.brackets[0] if self.brackets else None
        if not sample_bracket:
            return
        
        # First, create the bar chart visualization
        self._visualize_matchup_bars()
    
    def _visualize_matchup_bars(self):
        # Set a professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        
        
        # Split matchups into groups of 8 for better visualization
        matchup_items = list(self.matchup_stats.items())
        # print(f"Total matchups: {len(matchup_items)}")
        region1_r1 = matchup_items[0:8]
        region1_r2 = matchup_items[8:12]
        region1_r3 = matchup_items[12:14]
        region1_r4 = [matchup_items[14]]
        region2_r1 = matchup_items[15:23]
        region2_r2 = matchup_items[23:27]
        region2_r3 = matchup_items[27:29]
        region2_r4 = [matchup_items[29]]
        region3_r1 = matchup_items[30:38]
        region3_r2 = matchup_items[38:42]
        region3_r3 = matchup_items[42:44]
        region3_r4 = [matchup_items[44]]
        region4_r1 = matchup_items[45:53]
        region4_r2 = matchup_items[53:57]
        region4_r3 = matchup_items[57:59]
        region4_r4 = [matchup_items[59]]
        final_four = matchup_items[60:62]
        championship = matchup_items[62]

        r1_items = [region1_r1, region2_r1, region3_r1,  region4_r1]
        r2_items = [region1_r2, region2_r2, region3_r2, region4_r2]
        r3_items = [region1_r3, region2_r3, region3_r3, region4_r3]
        r4_items = [region1_r4, region2_r4, region3_r4, region4_r4]
        ff_items = [final_four]
        championship_items = [[championship]]
            
        # num_groups = (len(matchup_items) + 7) // 8
        
        # for group in range(num_groups):
        #     start_idx = group * 8
        #     end_idx = min(start_idx + 8, len(matchup_items))
        #     current_matchups = matchup_items[start_idx:end_idx]
            
            # Set up figure with better aesthetics
        regions = ['W', 'X', 'Y', 'Z']
        for idx, r1_item in enumerate(r1_items):
            self.matchup_visualization(r1_item, regions[idx], 1)

        for idx, r2_item in enumerate(r2_items):
            self.matchup_visualization(r2_item, regions[idx], 2)
        
        for idx, r3_item in enumerate(r3_items):
            # print(f"R3 Item: {r3_item}")
            self.matchup_visualization(r3_item, regions[idx], 3)
        
        for idx, r4_item in enumerate(r4_items):
            # print(f"R4 Item: {r4_item}")
            self.matchup_visualization(r4_item, regions[idx], 4)
        
        for idx, ff_item in enumerate(ff_items):
            self.matchup_visualization(ff_item, "Final Four", 5)

        for idx, champ_item in enumerate(championship_items):
            self.matchup_visualization(champ_item, "Championship", 6)
            

    def matchup_visualization(self, matchup_items, region, round_num):
        if not matchup_items:  # Skip if no matchups
            return
            
        region_name = {
                'W': 'West',
                'X': 'East',
                'Y': 'South',
                'Z': 'Midwest',
                'Final Four': 'Final Four',
                'Championship': 'Championship'
        }
        sub_dir = ["Round 64", "Round 32", "Sweet 16", "Elite 8", "", ""]

        plt.figure(figsize=(14, 14), facecolor='white')
        plt.subplots_adjust(hspace=0.5)

        # Define a better color palette
        team1_color = '#1e88e5'  # Blue
        team2_color = '#d81b60'  # Red
        
        for i, (matchup, stats) in enumerate(matchup_items):
                
            team1, team2 = matchup
            
            # Get team names with error handling
            team1_name = self.brackets[0].get_team_name(team1)
            team2_name = self.brackets[0].get_team_name(team2) 
            
            # Calculate percentages
            total = sum(stats.values())
            team1_pct = (stats[team1] / total) * 100 if team1 in stats else 0
            team2_pct = (stats[team2] / total) * 100 if team2 in stats else 0
            
            # Create a subplot for each matchup
            ax = plt.subplot(len(matchup_items), 1, i+1)
            
            # Add background shading for better visual separation
            ax.axhspan(-0.25, 0.25, color='#f5f5f5', zorder=0)
            
            # Create the horizontal bars with enhanced styling
            bars1 = plt.barh([0], [team1_pct], label=team1_name, color=team1_color, 
                                height=0.5, alpha=0.85, edgecolor='white', linewidth=1)
            bars2 = plt.barh([0], [team2_pct], left=[team1_pct], label=team2_name, 
                                color=team2_color, height=0.5, alpha=0.85, edgecolor='white', linewidth=1)
            
            # Add percentage labels with improved styling
            if team1_pct > 8:  # Only show label if bar is wide enough
                plt.text(team1_pct/2, 0, f'{team1_pct:.1f}%', 
                        ha='center', va='center', color='white', fontweight='bold', fontsize=11)
            if team2_pct > 8:
                plt.text(team1_pct + team2_pct/2, 0, f'{team2_pct:.1f}%', 
                        ha='center', va='center', color='white', fontweight='bold', fontsize=11)
            
            # Customize the subplot
            plt.yticks([])
            plt.xlim(0, 100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#dddddd')
            
            # Add small tick marks at 25%, 50%, 75%
            plt.xticks([25, 50, 75], ['25%', '50%', '75%'], color='#777777', fontsize=9)
            
            # Get seeds with error handling
            team1_seed = tourney_seeds[
                (tourney_seeds['Season'] == self.season) & 
                (tourney_seeds['TeamID'] == team1)
            ]['SeedNum'].iloc[0] 
            
            team2_seed = tourney_seeds[
                (tourney_seeds['Season'] == self.season) & 
                (tourney_seeds['TeamID'] == team2)
            ]['SeedNum'].iloc[0] 
            
            # Add a title that shows seeds with improved formatting
            plt.title(f'({team1_seed}) {team1_name} vs ({team2_seed}) {team2_name}', 
                        loc='left', pad=5, fontsize=12, fontweight='bold', color='#333333')
            
            # Add enhanced legend with team seeds
            if i == 0:  # Only show legend for the first subplot
                legend = plt.legend([f'({team1_seed}) {team1_name}', f'({team2_seed}) {team2_name}'],
                                    bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
                                    frameon=True, framealpha=0.9, edgecolor='#dddddd')
        
        # Add a more stylish title
        plt.suptitle(f'Tournament Matchup Win Probabilities - {region_name[region]} {sub_dir[round_num-1]}', 
                        fontsize=16, y=0.98, fontweight='bold', color='#333333')
        
        # Add a subtle footer with simulation info
        plt.figtext(0.5, 0.01, f'Based on {self.num_simulations} simulations', 
                    ha='center', fontsize=9, fontstyle='italic', color='#666666')
       
        plt.tight_layout()
        plt.savefig(f'MatchupResults{"/" + sub_dir[round_num-1] if sub_dir[round_num-1] else ""}/{region_name[region]} Predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    

class TournamentPredictor:
    def __init__(self, season, early_model=None, middle_model=None, final_model=None):
        self.season = season
        # Models are passed in after PyCaret selection
        self.early_model = early_model
        self.middle_model = middle_model
        self.final_model = final_model
        self.scaler = StandardScaler()
    
    def predict_proba(self, features, round_num):
        if round_num <= 2:
            # Early rounds: use features
            return {'early': self.early_model.predict_proba(features)}
        elif round_num <= 4:
            # Middle rounds: use balanced model
            return {'middle': self.middle_model.predict_proba(features)}
        else:
            # Final rounds: use detailed stats model
            return {'final': self.final_model.predict_proba(features)}
    

season_to_simulate = 2025  # Current season


# Create and train the tournament predictor with PyCaret models

predictor = TournamentPredictor(season_to_simulate, early_model, middle_model, final_model)


# Run tournament simulation with round-based predictions
print("\nTournament Predictions (Round-Based):")
simulation = BracketSimulation(season_to_simulate, predictor, scaler, num_simulations=100)
simulation.simulate_brackets()
simulation.visualize_matchup_stats()

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
from preprocessing import clean_ncaa_tourney_compact_results, clean_ncaa_tourney_detailed_results, \
    clean_regular_season_detailed_results, clean_tourney_seeds, clean_team_conferences

# Run preprocessing steps
# clean_ncaa_tourney_compact_results()
# clean_ncaa_tourney_detailed_results()
# clean_regular_season_detailed_results()
# clean_tourney_seeds()
# clean_team_conferences()

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
    match = re.search(r'(\d+)', seed_str)
    return int(match.group(1)) if match else np.nan

tourney_seeds['SeedNum'] = tourney_seeds['Seed'].apply(extract_seed)

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
            'Team1_SeedWin%': seed_results.loc[winner_feat['SeedNum'].astype(int), "WIN%"],
            'Team1_SeedR1%': seed_results.loc[winner_feat['SeedNum'].astype(int), "R64%"],
            'Team1_SeedR2%': seed_results.loc[winner_feat['SeedNum'].astype(int), "R32%"],
            'Team2_AvgPoints': loser_feat['AvgPoints'],
            'Team2_AvgOppPoints': loser_feat['AvgOppPoints'],
            'Team2_SeedNum': loser_feat['SeedNum'],
            'Team2_SeedWin%': seed_results.loc[loser_feat['SeedNum'].astype(int), "WIN%"],
            'Team2_SeedR1%': seed_results.loc[loser_feat['SeedNum'].astype(int), "R64%"],
            'Team2_SeedR2%': seed_results.loc[loser_feat['SeedNum'].astype(int), "R32%"],
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

# Evaluate early round models (using only seed features)
# X_train_scaled['Team1_R1%'] = seed_results.loc[X_train_scaled['Team1_SeedNum'].astype(int), "R64%"]
# X_train_scaled['Team1_R2%'] = seed_results.loc[X_train_scaled['Team1_SeedNum'].astype(int), "R32%"]
# X_train_scaled['Team2_R1%'] = seed_results.loc[X_train_scaled['Team2_SeedNum'].astype(int), "R64%"]
# X_train_scaled['Team2_R2%'] = seed_results.loc[X_train_scaled['Team2_SeedNum'].astype(int), "R32%"]
# X_test_scaled['Team1_R1%'] = seed_results.loc[X_test_scaled['Team1_SeedNum'].astype(int), "R64%"]
# X_test_scaled['Team1_R2%'] = seed_results.loc[X_test_scaled['Team1_SeedNum'].astype(int), "R32%"]
# X_test_scaled['Team2_R1%'] = seed_results.loc[X_test_scaled['Team2_SeedNum'].astype(int), "R64%"]
# X_test_scaled['Team2_R2%'] = seed_results.loc[X_test_scaled['Team2_SeedNum'].astype(int), "R32%"]

early_model = evaluate_models(early_models, X_train_scaled, X_test_scaled, 
                            y_train, y_test, "Early Round")

for index, row in X_train.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "S16%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "E8%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "S16%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "E8%"]

# X_train['Team1_R1%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "S16%"]
# X_train['Team1_R2%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "E8%"]
# X_train['Team2_R1%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "S16%"]
# X_train['Team2_R2%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "E8%"]

for index, row in X_test.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "S16%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "E8%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "S16%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "E8%"]

# X_test['Team1_R1%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "S16%"]
# X_test['Team1_R2%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "E8%"]
# X_test['Team2_R1%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "S16%"]
# X_test['Team2_R2%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "E8%"]

# Evaluate middle round models (using all features)
middle_model = evaluate_models(middle_models, X_train_scaled, X_test_scaled, 
                              y_train, y_test, "Middle Round")

for _, row in X_train.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F4%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F2%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F4%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F2%"]
        

# X_train['Team1_R1%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "F4%"]
# X_train['Team1_R2%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "F2%"]
# X_train['Team2_R1%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "F4%"]
# X_train['Team2_R2%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "F2%"]

for _, row in X_test.iterrows():
    row['Team1_SeedR1%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F4%"]
    row['Team1_SeedR2%'] = seed_results.at[row['Team1_SeedNum'].astype(int), "F2%"]
    row['Team2_SeedR1%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F4%"]
    row['Team2_SeedR2%'] = seed_results.at[row['Team2_SeedNum'].astype(int), "F2%"]

# X_test['Team1_R1%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "F4%"]
# X_test['Team1_R2%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "F2%"]
# X_test['Team2_R1%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "F4%"]
# X_test['Team2_R2%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "F2%"]

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
            return None
        
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
                print(f"Team1: {team1}")
                team2 = next((t for t in teams if t[1] == seed2), None)
                print(f"Team2: {team2}")
                
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
        
        # Then create the bracket visualization
        self._visualize_tournament_bracket()
    
    def _visualize_matchup_bars(self):
        # Set a professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        
        
        # Split matchups into groups of 8 for better visualization
        matchup_items = list(self.matchup_stats.items())
        print(f"Total matchups: {len(matchup_items)}")
        region1_r1 = matchup_items[0:8]
        region1_r2 = matchup_items[8:12]
        region1_r3 = matchup_items[12:14]
        region1_r4 = matchup_items[14]
        region2_r1 = matchup_items[15:23]
        region2_r2 = matchup_items[23:27]
        region2_r3 = matchup_items[27:29]
        region2_r4 = matchup_items[29]
        region3_r1 = matchup_items[30:38]
        region3_r2 = matchup_items[38:42]
        region3_r3 = matchup_items[42:44]
        region3_r4 = matchup_items[44]
        region4_r1 = matchup_items[45:53]
        region4_r2 = matchup_items[53:57]
        region4_r3 = matchup_items[57:59]
        region4_r4 = matchup_items[59]
        final_four = matchup_items[60:62]
        championship = matchup_items[62]

        r1_items = [region1_r1, region2_r1, region3_r1,  region4_r1]
        r2_items = [region1_r2, region2_r2, region3_r2, region4_r2]
        r3_items = [region1_r3, region2_r3, region3_r3, region4_r3]
        r4_items = [region1_r4, region2_r4, region3_r4, region4_r4]
        ff_items = final_four
        championship_items = championship
            
        # num_groups = (len(matchup_items) + 7) // 8
        
        # for group in range(num_groups):
        #     start_idx = group * 8
        #     end_idx = min(start_idx + 8, len(matchup_items))
        #     current_matchups = matchup_items[start_idx:end_idx]
            
            # Set up figure with better aesthetics
        regions = ['W', 'X', 'Y', 'Z']
        for idx, r1_item in enumerate(r1_items):
            self.matchup_visualization(r1_item, regions[idx], round_num=1)

        for idx, r2_item in enumerate(r2_items):
            self.matchup_visualization(r2_item, regions[idx], round_num=2)
        
        for idx, r3_item in enumerate(r3_items):
            self.matchup_visualization(r3_item, regions[idx], round_num=3)
        
        for idx, r4_item in enumerate(r4_items):
            self.matchup_visualization(r4_item, regions[idx], round_num=4)
        
        for idx, ff_item in enumerate(ff_items):
            self.matchup_visualization(ff_item, "Final Four")

        for idx, champ_item in enumerate(championship_items):
            self.matchup_visualization(champ_item, "Championship")
            

    def matchup_visualization(self, matchup_items, region, round_num=""):
        plt.figure(figsize=(14, 14), facecolor='white')
        plt.subplots_adjust(hspace=0.5)

        # Define a better color palette
        team1_color = '#1e88e5'  # Blue
        team2_color = '#d81b60'  # Red
        
        for i, (matchup, stats) in enumerate(matchup_items):
            team1, team2 = matchup
            team1_name = self.brackets[0].get_team_name(team1)
            team2_name = self.brackets[0].get_team_name(team2)
            
            total = sum(stats.values())
            team1_pct = (stats[team1] / total) * 100
            team2_pct = (stats[team2] / total) * 100
            
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
            
            # Get seeds for both teams
            team1_seed = tourney_seeds[
                (tourney_seeds['Season'] == self.season) & 
                (tourney_seeds['TeamID'] == team1)
            ]['SeedNum'].iloc[0] if not tourney_seeds.empty else '?'
            
            team2_seed = tourney_seeds[
                (tourney_seeds['Season'] == self.season) & 
                (tourney_seeds['TeamID'] == team2)
            ]['SeedNum'].iloc[0] if not tourney_seeds.empty else '?'
            
            # Add a title that shows seeds with improved formatting
            plt.title(f'({team1_seed}) {team1_name} vs ({team2_seed}) {team2_name}', 
                        loc='left', pad=5, fontsize=12, fontweight='bold', color='#333333')
            
            # Add enhanced legend with team seeds
            if i == 0:  # Only show legend for the first subplot
                legend = plt.legend([f'({team1_seed}) {team1_name}', f'({team2_seed}) {team2_name}'],
                                    bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
                                    frameon=True, framealpha=0.9, edgecolor='#dddddd')
        
        # Add a more stylish title
        plt.suptitle(f'Tournament Matchup Win Probabilities - {region} {round_num}', 
                        fontsize=16, y=0.98, fontweight='bold', color='#333333')
        
        # Add a subtle footer with simulation info
        plt.figtext(0.5, 0.01, f'Based on {self.num_simulations} simulations', 
                    ha='center', fontsize=9, fontstyle='italic', color='#666666')
        
        plt.tight_layout()
        plt.savefig(f'MatchupResults/matchup_predictions_group{str(round_num) + "_" + region}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # def _visualize_tournament_bracket(self):
    #     # Set a clean, professional aesthetic
    #     plt.style.use('seaborn-v0_8-whitegrid')
        
    #     # Create a figure with appropriate dimensions and white background
    #     plt.figure(figsize=(24, 30), facecolor='white')
        
    #     # Define the number of rounds and teams per round - updated for full 64-team tournament
    #     rounds = ['First Round', 'Second Round', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
    #     teams_per_region_round = [16, 8, 4, 2, 1, 0]  # Teams per region in each round (total: 64, 32, 16, 8, 4, 2)
        
    #     # Define colors for regions and teams - using a more harmonious color palette
    #     region_colors = {
    #         'W': '#2980b9',  # West - Belize Hole Blue
    #         'X': '#27ae60',  # East - Nephritis Green
    #         'Y': '#c0392b',  # South - Pomegranate Red
    #         'Z': '#8e44ad'   # Midwest - Wisteria Purple
    #     }
    #     bg_color = '#ffffff'     # Pure white background
    #     line_color = '#e0e0e0'   # Very light gray for grid lines
    #     text_color = '#2c3e50'   # Midnight Blue for text
    #     champion_color = '#f39c12'  # Orange for champion
        
    #     # Add a subtle background
    #     plt.gca().patch.set_facecolor(bg_color)
        
    #     # Add title with season and simulation count info
    #     plt.suptitle(f"NCAA Tournament {self.season} - Bracket Predictions\n(Based on {self.num_simulations} simulations)", 
    #               fontsize=20, fontweight='bold', y=0.98, color=text_color)
        
    #     # Improve spacing between regions
    #     region_spacing = 0.18  # Reduced spacing between regions (as fraction of figure height)
    #     region_height = (1.0 - 3*region_spacing) / 4  # Height of each region
        
    #     # Define better positions for each region - more evenly distributed
    #     round_positions = {}
    #     region_y_positions = {
    #         'W': 0.85,                                       # West at top
    #         'X': 0.85 - (region_height + region_spacing),     # East below West
    #         'Y': 0.85 - 2*(region_height + region_spacing),   # South below East
    #         'Z': 0.85 - 3*(region_height + region_spacing)    # Midwest at bottom
    #     }
        
    #     # For each region, calculate team positions in each round with proper spacing for 16 teams
    #     for region_code, region_top in region_y_positions.items():
    #         round_positions[region_code] = {}
            
    #         # Calculate regional rounds positions (rounds 1-4) to fit all teams
    #         for round_idx in range(4):  # Regional rounds only
    #             # Number of teams in each round per region (16, 8, 4, 2)
    #             num_teams = teams_per_region_round[round_idx]
                
    #             # Adjust spacing to fit all teams properly
    #             # First round needs to fit 16 teams per region
    #             spacing = (region_height * 0.9) / (num_teams - 1) if num_teams > 1 else 0
    #             y_positions = [region_top - i*spacing for i in range(num_teams)]
    #             round_positions[region_code][round_idx] = y_positions
                
    #         # Print the number of team positions in the first round for this region
    #         print(f"{region_code} region first round positions: {len(round_positions[region_code][0])}")
        
    #     # Improve positioning for Final Four and Championship (centered)
    #     final_four_y = 0.40
    #     championship_y = 0.25
    #     final_positions = {
    #         4: [final_four_y, final_four_y - 0.10],  # Final Four - better spacing
    #         5: [championship_y]                      # Championship
    #     }
        
    #     # Draw regional markers
    #     for region_code, region_top in region_y_positions.items():
    #         region_name = {
    #             'W': 'WEST',
    #             'X': 'EAST',
    #             'Y': 'SOUTH',
    #             'Z': 'MIDWEST'
    #         }[region_code]
    #         plt.text(0.02, region_top + 0.02, region_name, 
    #                 fontsize=16, fontweight='bold', color=region_colors[region_code])
        
    #     # Draw grid lines to separate rounds
    #     for round_idx in range(6):
    #         x_pos = 0.05 + round_idx * 0.15  # Horizontal position of round
    #         plt.axvline(x_pos, color=line_color, linestyle='-', alpha=0.3, zorder=0)
    #         # Add round labels at top
    #         plt.text(x_pos + 0.075, 0.97, rounds[round_idx], 
    #                 fontsize=12, ha='center', va='center', fontweight='bold')
        
    #     # Draw regional brackets (Rounds 1-4)
    #     for region_code in ['W', 'X', 'Y', 'Z']:
    #         # Process each round within the region
    #         for round_idx in range(4):  # Regional rounds 1-4
    #             x_start = 0.05 + round_idx * 0.15  # Start position of current round
    #             x_end = x_start + 0.15             # End position (next round)
                
    #             # Get matchups for this region and round
    #             round_matchups = [m for m in self.bracket_results[round_idx+1] 
    #                             if m[0][0] == region_code]  # Filter by region
                
    #             # Draw each matchup
    #             for i, matchup in enumerate(round_matchups):
    #                 team1_info, team2_info, winner_info = matchup
    #                 _, team1_id, team1_seed = team1_info
    #                 _, team2_id, team2_seed = team2_info
    #                 _, winner_id = winner_info
                    
    #                 # Get y-coordinates
    #                 if round_idx < 4:  # Regional rounds
    #                     y_positions = round_positions[region_code][round_idx]
    #                     idx1 = i * 2
    #                     idx2 = i * 2 + 1
    #                     y1 = y_positions[idx1] if idx1 < len(y_positions) else 0
    #                     y2 = y_positions[idx2] if idx2 < len(y_positions) else 0
                        
    #                     # Position in next round
    #                     if round_idx < 3:  # Not Elite 8
    #                         next_y_positions = round_positions[region_code][round_idx+1]
    #                         y_next = next_y_positions[i] if i < len(next_y_positions) else 0
    #                     else:  # Elite 8 to Final Four transition
    #                         # Find which semifinal this region feeds into
    #                         semifinal_idx = 0 if region_code in ['W', 'Y'] else 1
    #                         y_next = final_positions[4][semifinal_idx]
                    
    #                 # Get team names
    #                 team1_name = self.brackets[0].get_team_name(team1_id)
    #                 team2_name = self.brackets[0].get_team_name(team2_id)
                    
    #                 # Truncate long team names
    #                 if len(team1_name) > 14:
    #                     team1_name = team1_name[:12] + '...'
    #                 if len(team2_name) > 14:
    #                     team2_name = team2_name[:12] + '...'
                    
    #                 # Get matchup stats for win percentages
    #                 if (team1_id, team2_id) in self.matchup_stats:
    #                     stats = self.matchup_stats[(team1_id, team2_id)]
    #                 elif (team2_id, team1_id) in self.matchup_stats:
    #                     stats = self.matchup_stats[(team2_id, team1_id)]
    #                 else:
    #                     stats = {team1_id: 50, team2_id: 50}  # Default if not found
                    
    #                 total = sum(stats.values())
    #                 team1_pct = (stats[team1_id] / total) * 100
    #                 team2_pct = (stats[team2_id] / total) * 100
                    
    #                 # Draw connector to next round
    #                 if round_idx < 4:  # Only for rounds leading to another round
    #                     plt.plot([x_end-0.03, x_end], [y_next, y_next], 
    #                             color='#aaaaaa', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
                    
    #                 # Draw connecting lines with thickness based on win probability
    #                 team1_color = region_colors[region_code]
    #                 team2_color = region_colors[region_code]
    #                 winner_color = region_colors[region_code]
                    
    #                 # Draw team connecting lines
    #                 if round_idx < 4:  # Only for rounds leading to another round
    #                     # Team 1 line
    #                     plt.plot([x_start, x_end-0.03], [y1, y_next], 
    #                             color=team1_color, alpha=0.2+0.8*team1_pct/100, 
    #                             linewidth=1.0+2.0*team1_pct/100, solid_capstyle='round', zorder=2)
                        
    #                     # Team 2 line
    #                     plt.plot([x_start, x_end-0.03], [y2, y_next], 
    #                             color=team2_color, alpha=0.2+0.8*team2_pct/100, 
    #                             linewidth=1.0+2.0*team2_pct/100, solid_capstyle='round', zorder=2)
                    
    #                 # Team boxes
    #                 team1_box = dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                                alpha=0.9, edgecolor=team1_color, linewidth=1.5)
    #                 team2_box = dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                                alpha=0.9, edgecolor=team2_color, linewidth=1.5)
                    
    #                 # Highlight winner
    #                 winner_box = team1_box if winner_id == team1_id else team2_box
    #                 winner_box['edgecolor'] = winner_color
    #                 winner_box['linewidth'] = 2.0
                    
    #                 # Add team names with seeds
                
    #             # Get team names
    #             team1_name = self.brackets[0].get_team_name(team1_id)
    #             team2_name = self.brackets[0].get_team_name(team2_id)
                
    #             # Truncate long team names
    #             if len(team1_name) > 14:
    #                 team1_name = team1_name[:12] + '...'
    #             if len(team2_name) > 14:
    #                 team2_name = team2_name[:12] + '...'
                
    #             # Get matchup stats for win percentages
    #             if (team1_id, team2_id) in self.matchup_stats:
    #                 stats = self.matchup_stats[(team1_id, team2_id)]
    #             elif (team2_id, team1_id) in self.matchup_stats:
    #                 stats = self.matchup_stats[(team2_id, team1_id)]
    #             else:
    #                 stats = {team1_id: 50, team2_id: 50}  # Default if not found
                
    #             total = sum(stats.values())
    #             team1_pct = (stats[team1_id] / total) * 100
    #             team2_pct = (stats[team2_id] / total) * 100
                
    #             # Draw connector to next round with improved styling
    #             if round_idx < 4:  # Only for rounds leading to another round
    #                 plt.plot([x_end-0.03, x_end], [y_next, y_next], 
    #                         color='#cccccc', linestyle='-', linewidth=1.0, alpha=0.6, zorder=1)
                
    #             # Use consistent colors with improved opacity handling
    #             team1_color = region_colors[region_code]
    #             team2_color = region_colors[region_code]
    #             winner_color = region_colors[region_code]
                
    #             # Draw team connecting lines with better styling
    #             if round_idx < 4:  # Only for rounds leading to another round
    #                 # Team 1 line - more subtle gradient based on win probability
    #                 plt.plot([x_start, x_end-0.03], [y1, y_next], 
    #                         color=team1_color, alpha=0.3+0.6*team1_pct/100, 
    #                         linewidth=0.8+1.5*team1_pct/100, solid_capstyle='round', zorder=2)
                    
    #                 # Team 2 line - more subtle gradient based on win probability
    #                 plt.plot([x_start, x_end-0.03], [y2, y_next], 
    #                         color=team2_color, alpha=0.3+0.6*team2_pct/100, 
    #                         linewidth=0.8+1.5*team2_pct/100, solid_capstyle='round', zorder=2)
                
    #             # Improved team boxes with cleaner styling
    #             team1_box = dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                            alpha=0.95, edgecolor=team1_color, linewidth=1.0)
    #             team2_box = dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                            alpha=0.95, edgecolor=team2_color, linewidth=1.0)
                
    #             # Highlight winner with bolder style
    #             winner_box = team1_box if winner_id == team1_id else team2_box
    #             winner_box['edgecolor'] = winner_color
    #             winner_box['linewidth'] = 1.8
    #             winner_box['boxstyle'] = 'round,pad=0.3'
                
    #             # Add team names with seeds and cleaner fonts
    #             plt.text(x_start - 0.04, y1, f"({team1_seed}) {team1_name}", 
    #                    fontsize=9 if round_idx <= 1 else 10, ha='right', va='center', bbox=team1_box,
    #                    fontweight='normal' if winner_id != team1_id else 'bold')
                
    #             plt.text(x_start - 0.04, y2, f"({team2_seed}) {team2_name}", 
    #                    fontsize=9 if round_idx <= 1 else 10, ha='right', va='center', bbox=team2_box,
    #                    fontweight='normal' if winner_id != team2_id else 'bold')
        
    #     # Draw Final Four (Round 5)
    #     if len(self.bracket_results[5]) == 2:  # Make sure we have Final Four results
    #         x_start = 0.05 + 4 * 0.15  # Final Four x position
    #         x_end = x_start + 0.15      # Championship x position
            
    #         # First semifinal
    #         semifinal1 = self.bracket_results[5][0]
    #         team1_region, team1_id, team1_seed = semifinal1[0]
    #         team2_region, team2_id, team2_seed = semifinal1[1]
    #         winner_region, winner_id = semifinal1[2]
            
    #         # Get y positions
    #         y1 = final_positions[4][0]  # First finalist position
    #         y_next = final_positions[5][0]  # Championship position
            
    #         # Get team names
    #         team1_name = self.brackets[0].get_team_name(team1_id)
    #         team2_name = self.brackets[0].get_team_name(team2_id)
            
    #         # Truncate long team names
    #         if len(team1_name) > 14:
    #             team1_name = team1_name[:12] + '...'
    #         if len(team2_name) > 14:
    #             team2_name = team2_name[:12] + '...'
            
    #         # Get matchup stats
    #         if (team1_id, team2_id) in self.matchup_stats:
    #             stats = self.matchup_stats[(team1_id, team2_id)]
    #         elif (team2_id, team1_id) in self.matchup_stats:
    #             stats = self.matchup_stats[(team2_id, team1_id)]
    #         else:
    #             stats = {team1_id: 50, team2_id: 50}
                
    #         total = sum(stats.values())
    #         team1_pct = (stats[team1_id] / total) * 100
    #         team2_pct = (stats[team2_id] / total) * 100
            
    #         # Draw connector to championship
    #         plt.plot([x_end-0.03, x_end], [y_next, y_next], 
    #                 color='#aaaaaa', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
            
    #         # Draw team connecting lines
    #         team1_color = region_colors[team1_region]
    #         team2_color = region_colors[team2_region]
            
    #         # Team 1 line
    #         plt.plot([x_start, x_end-0.03], [y1, y_next], 
    #                 color=team1_color, alpha=0.2+0.8*team1_pct/100, 
    #                 linewidth=1.0+2.0*team1_pct/100, solid_capstyle='round', zorder=2)
            
    #         # Create team boxes
    #         team1_box = dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                        alpha=0.9, edgecolor=team1_color, linewidth=1.5)
            
    #         # Highlight winner
    #         if winner_id == team1_id:
    #             team1_box['edgecolor'] = team1_color
    #             team1_box['linewidth'] = 2.0
            
    #         # Add team name with seed and region
    #         plt.text(x_start - 0.04, y1, f"({team1_seed}) {team1_name} [{team1_region}]", 
    #                fontsize=9, ha='right', va='center', bbox=team1_box)
            
    #         # Second semifinal
    #         semifinal2 = self.bracket_results[5][1]
    #         team1_region, team1_id, team1_seed = semifinal2[0]
    #         team2_region, team2_id, team2_seed = semifinal2[1]
    #         winner_region, winner_id = semifinal2[2]
            
    #         # Get y positions
    #         y2 = final_positions[4][1]  # Second finalist position
            
    #         # Get team names
    #         team1_name = self.brackets[0].get_team_name(team1_id)
    #         team2_name = self.brackets[0].get_team_name(team2_id)
            
    #         # Truncate long team names
    #         if len(team1_name) > 14:
    #             team1_name = team1_name[:12] + '...'
    #         if len(team2_name) > 14:
    #             team2_name = team2_name[:12] + '...'
            
    #         # Get matchup stats
    #         if (team1_id, team2_id) in self.matchup_stats:
    #             stats = self.matchup_stats[(team1_id, team2_id)]
    #         elif (team2_id, team1_id) in self.matchup_stats:
    #             stats = self.matchup_stats[(team2_id, team1_id)]
    #         else:
    #             stats = {team1_id: 50, team2_id: 50}
                
    #         total = sum(stats.values())
    #         team1_pct = (stats[team1_id] / total) * 100
    #         team2_pct = (stats[team2_id] / total) * 100
            
    #         # Draw team connecting lines
    #         team1_color = region_colors[team1_region]
    #         team2_color = region_colors[team2_region]
            
    #         # Team 2 line
    #         plt.plot([x_start, x_end-0.03], [y2, y_next], 
    #                 color=team2_color, alpha=0.2+0.8*team2_pct/100, 
    #                 linewidth=1.0+2.0*team2_pct/100, solid_capstyle='round', zorder=2)
            
    #         # Create team boxes
    #         team2_box = dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                        alpha=0.9, edgecolor=team2_color, linewidth=1.5)
            
    #         # Highlight winner
    #         if winner_id == team2_id:
    #             team2_box['edgecolor'] = team2_color
    #             team2_box['linewidth'] = 2.0
            
    #         # Add team name with seed and region
    #         plt.text(x_start - 0.04, y2, f"({team2_seed}) {team2_name} [{team2_region}]", 
    #                fontsize=9, ha='right', va='center', bbox=team2_box)
        
    #     # Draw Championship (Round 6)
    #     if len(self.bracket_results[6]) == 1:  # Make sure we have Championship results
    #         champ_matchup = self.bracket_results[6][0]
    #         team1_region, team1_id, team1_seed = champ_matchup[0]
    #         team2_region, team2_id, team2_seed = champ_matchup[1]
    #         champion_region, champion_id = champ_matchup[2]
            
    #         # Get champion name
    #         champion_name = self.brackets[0].get_team_name(champion_id)
            
    #         # Create champion box
    #         champion_box = dict(boxstyle='round,pad=0.5', facecolor='#fffde7', 
    #                          alpha=0.95, edgecolor=champion_color, linewidth=2.5)
            
    #         # Add champion with larger font
    #         plt.text(0.05 + 5 * 0.15 + 0.075, championship_y - 0.05, 
    #                f"CHAMPION:\n({self.champion[1]}) {champion_name}\n[{champion_region} Region]", 
    #                fontsize=14, fontweight='bold', ha='center', va='center', 
    #                bbox=champion_box, color='#333333')
        
    #     # Round labels are already added at the top of the visualization
        
    #     # The plot limits are handled by tight_layout
        
    #     # Title is already added as suptitle at the beginning
    #     # The tournament info is already included in the title
        
    #     # Hide the axes
    #     plt.axis('off')
        
    #     # Add NCAA March Madness logo placeholder
    #     plt.figtext(0.5, 0.01, 'NCAA March Madness', ha='center', fontsize=14, 
    #                 fontweight='bold', color='#FF8800')
        
    #     # Save the plot with higher resolution
    #     plt.savefig('tournament_bracket.png', dpi=300, bbox_inches='tight')
    #     plt.close()

class TournamentPredictor:
    def __init__(self, season, early_model=None, middle_model=None, final_model=None):
        self.season = season
        # Models are passed in after PyCaret selection
        self.early_model = early_model
        self.middle_model = middle_model
        self.final_model = final_model
        self.scaler = StandardScaler()
    
    # def train(self, X_train, y_train, X_test, y_test, round_info):
    #     # Extract seed features for early round mode
        
    #     # Evaluate early round model
    #     early_preds = self.early_model.predict(X_test)
    #     early_score = accuracy_score(y_test, early_preds)
    #     print(f"Early Round Model Score: {early_score}")
    #     print(classification_report(y_test, early_preds))

    #     for index, row in X_train.iterrows():
    #         row['Team1_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "S16%"]
    #         row['Team1_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "E8%"]
    #         row['Team2_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "S16%"]
    #         row['Team2_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "E8%"]

    #     # X_train['Team1_R1%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "S16%"]
    #     # X_train['Team1_R2%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "E8%"]
    #     # X_train['Team2_R1%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "S16%"]
    #     # X_train['Team2_R2%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "E8%"]

    #     for index, row in X_test.iterrows():
    #         row['Team1_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "S16%"]
    #         row['Team1_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "E8%"]
    #         row['Team2_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "S16%"]
    #         row['Team2_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "E8%"]
        
    #     # X_test['Team1_R1%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "S16%"]
    #     # X_test['Team1_R2%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "E8%"]
    #     # X_test['Team2_R1%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "S16%"]
    #     # X_test['Team2_R2%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "E8%"]
        
    #     # Evaluate middle round model
    #     middle_preds = self.middle_model.predict(X_test)
    #     middle_score = accuracy_score(y_test, middle_preds)
    #     print(f"Middle Round Model Score: {middle_score}")
    #     print(classification_report(y_test, middle_preds))

    #     for index, row in X_train.iterrows():
    #         row['Team1_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "F4%"]
    #         row['Team1_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "F2%"]
    #         row['Team2_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "F4%"]
    #         row['Team2_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "F2%"]
        

    #     # X_train['Team1_R1%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "F4%"]
    #     # X_train['Team1_R2%'] = seed_results.loc[X_train['Team1_SeedNum'].astype(int), "F2%"]
    #     # X_train['Team2_R1%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "F4%"]
    #     # X_train['Team2_R2%'] = seed_results.loc[X_train['Team2_SeedNum'].astype(int), "F2%"]

    #     for index, row in X_test.iterrows():
    #         row['Team1_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "F4%"]
    #         row['Team1_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team1_SeedNum'].astype(int), "F2%"]
    #         row['Team2_SeedR1%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "F4%"]
    #         row['Team2_SeedR2%'] = seed_results.at[training_df.loc[index, 'Team2_SeedNum'].astype(int), "F2%"]

    #     # X_test['Team1_R1%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "F4%"]
    #     # X_test['Team1_R2%'] = seed_results.loc[X_test['Team1_SeedNum'].astype(int), "F2%"]
    #     # X_test['Team2_R1%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "F4%"]
    #     # X_test['Team2_R2%'] = seed_results.loc[X_test['Team2_SeedNum'].astype(int), "F2%"]
        
    #     # Evaluate final round model
    #     final_preds = self.final_model.predict(X_test)
    #     final_score = accuracy_score(y_test, final_preds)
    #     print(f"Final Round Model Score: {final_score}")
    #     print(classification_report(y_test, final_preds))
    
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
    
    def _get_seed_features(self, X):
        # Create features based on seed differences and historical upset patterns
        seed_features = []
        for i in range(len(X)):
            # Extract team seeds directly from the scaled features
            team1_seed = X[i, 2] if isinstance(X, np.ndarray) else X.iloc[i]['Team1_SeedNum']
            team2_seed = X[i, 5] if isinstance(X, np.ndarray) else X.iloc[i]['Team2_SeedNum']
            
            # Features that capture tournament dynamics
            seed_diff = abs(team1_seed - team2_seed)
            seed_product = team1_seed * team2_seed
            seed_ratio = max(team1_seed, team2_seed) / min(team1_seed, team2_seed)
            upset_potential = 1 / (seed_diff + 1)  # Higher for closer seeds
            favorite_seed = min(team1_seed, team2_seed)
            underdog_seed = max(team1_seed, team2_seed)
            
            # Combine features
            seed_features.append([
                seed_diff,          # Raw difference in seeds
                seed_product,       # Product captures matchup difficulty
                seed_ratio,         # Relative seed strength
                upset_potential,    # Likelihood of an upset
                favorite_seed,      # Seed of the favored team
                underdog_seed       # Seed of the underdog team
            ])
        
        return np.array(seed_features)

# Load round information for training
tourney_rounds = pd.read_csv(os.path.join(project_root, 'MarchMadnessData', 'MNCAATourneyCompactResults.csv'))
season_to_simulate = 2024  # Current season
if season_to_simulate != 2021:
    tourney_rounds['Round'] = (tourney_rounds['DayNum'] - 134) // 2 + 1  # Convert DayNum to tournament round
else:
    tourney_rounds['Round'] = (tourney_rounds['DayNum'] - 136) // 2 + 1

# Create and train the tournament predictor with PyCaret models

predictor = TournamentPredictor(season_to_simulate, early_model, middle_model, final_model)
# predictor.train(X_train_scaled, y_train, X_test_scaled, y_test, tourney_rounds)

# Run tournament simulation with round-based predictions
print("\nTournament Predictions (Round-Based):")
simulation = BracketSimulation(season_to_simulate, predictor, scaler, num_simulations=100)
simulation.simulate_brackets()
simulation.visualize_matchup_stats()

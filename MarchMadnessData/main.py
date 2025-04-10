import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb

from pycaret.classification import *

# =============================================================================
# 1. Data Collection and Data Cleaning
# =============================================================================

# Load additional NCAA datasets
team_spellings = pd.read_csv('MTeamSpellings.csv')
tourney_seeds = pd.read_csv('MNCAATourneySeeds.csv')
team_conferences = pd.read_csv('MTeamConferences.csv')
reg_results = pd.read_csv('MRegularSeasonDetailedResults.csv')
tourney_results = pd.read_csv('MNCAATourneyCompactResults.csv')

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
        off_diff = winner_feat['AvgPoints'] - loser_feat['AvgPoints']
        def_diff = winner_feat['AvgOppPoints'] - loser_feat['AvgOppPoints']
        seed_diff = loser_feat['SeedNum'] - winner_feat['SeedNum']  # positive value: winner had a better (lower) seed
        
        training_rows.append({
            'OffEff_Diff': off_diff,
            'DefEff_Diff': def_diff,
            'Seed_Diff': seed_diff,
            'Target': 1
        })
        training_rows.append({
            'OffEff_Diff': -off_diff,
            'DefEff_Diff': -def_diff,
            'Seed_Diff': -seed_diff,
            'Target': 0
        })

training_df = pd.DataFrame(training_rows)
print("Training dataset shape:", training_df.shape)
print(training_df.head())

s = setup(training_df, target='Target', session_id=123)

# Compare models
best_model = compare_models()

# =============================================================================
# 3. Model Training & Evaluation
# =============================================================================
X = training_df.drop('Target', axis=1)
y = training_df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- Neural Network (MLPClassifier) ---
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
print("\nNeural Network (MLP) Accuracy:", accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("\nXGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# =============================================================================
# 4. Simulation and Bracket Prediction
# =============================================================================
def simulate_matchup(season, team1_id, team2_id, model, scaler_obj=None):
    team1_feat = get_team_features(season, team1_id)
    team2_feat = get_team_features(season, team2_id)
    if team1_feat is None or team2_feat is None:
        print(f"Missing features for team {team1_id} or {team2_id} in season {season}")
        return None
    
    off_diff = team1_feat['AvgPoints'] - team2_feat['AvgPoints']
    def_diff = team1_feat['AvgOppPoints'] - team2_feat['AvgOppPoints']
    seed_diff = team2_feat['SeedNum'] - team1_feat['SeedNum']
    
    features = np.array([[off_diff, def_diff, seed_diff]])
    if scaler_obj is not None:
        features = scaler_obj.transform(features)
    
    prediction = model.predict(features)
    return team1_id if prediction[0] == 1 else team2_id

# Example simulation of a round of matchups using the XGBoost model
sample_matchups = [
    (201, 202),
    (203, 204),
    (205, 206),
    (207, 208)
]
season_to_simulate = 2019  # Adjust this to a season available in your data

print("\nSample Bracket Simulation (using XGBoost):")
for matchup in sample_matchups:
    winner = simulate_matchup(season_to_simulate, matchup[0], matchup[1], xgb_model, scaler_obj=scaler)
    if winner is not None:
        print(f"Matchup {matchup}: Winner predicted is Team {winner}")

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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
clean_ncaa_tourney_compact_results()
clean_ncaa_tourney_detailed_results()
clean_regular_season_detailed_results()
clean_tourney_seeds()
clean_team_conferences()

# Load preprocessed NCAA datasets
data_dir = os.path.join(project_root, 'MarchMadnessData')
team_spellings = pd.read_csv(f'{data_dir}/MTeamSpellings.csv')
tourney_seeds = pd.read_csv(f'{data_dir}/MNCAATourneySeeds.csv')
team_conferences = pd.read_csv(f'{data_dir}/MTeamConferences.csv')
reg_results = pd.read_csv(f'{data_dir}/MRegularSeasonDetailedResults.csv')
tourney_results = pd.read_csv(f'{data_dir}/MNCAATourneyCompactResults.csv')

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
            'Team2_AvgPoints': loser_feat['AvgPoints'],
            'Team2_AvgOppPoints': loser_feat['AvgOppPoints'],
            'Team2_SeedNum': loser_feat['SeedNum'],
            'Target': 1
        })
        
        training_rows.append({
            'Team1ID': loser_id,
            'Team2ID': winner_id,
            'Team1_AvgPoints': loser_feat['AvgPoints'],
            'Team1_AvgOppPoints': loser_feat['AvgOppPoints'],
            'Team1_SeedNum': loser_feat['SeedNum'],
            'Team2_AvgPoints': winner_feat['AvgPoints'],
            'Team2_AvgOppPoints': winner_feat['AvgOppPoints'],
            'Team2_SeedNum': winner_feat['SeedNum'],
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
                'Team2_AvgPoints', 'Team2_AvgOppPoints', 'Team2_SeedNum']
features = training_df[feature_cols]
labels = training_df['Target']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Define model candidates for each round
early_models = {
    'logistic': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'svm': SVC(probability=True, random_state=42)
}

middle_models = {
    'gradient_boost': GradientBoostingClassifier(random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=150, random_state=42),
    'neural_net': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
}

final_models = {
    'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'gradient_boost': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'neural_net': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Function to evaluate models
def evaluate_models(models, X_train, X_test, y_train, y_test, round_name):
    print(f"\nEvaluating {round_name} models:")
    best_score = 0
    best_model = None
    accuracies = []
    model_names = []
    
    plt.figure(figsize=(12, 6))
    
    for name, model in models.items():
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        accuracies.append(score)
        model_names.append(name.upper())
        
        print(f"\n{name.upper()} Results:")
        print(f"Accuracy: {score:.4f}")
        print(classification_report(y_test, y_pred))
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Create bar plot of accuracies
    plt.bar(model_names, accuracies)
    plt.title(f'Model Accuracies for {round_name}')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 0.8)  # Set y-axis range for better visualization
    
    # Add value labels on top of each bar
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, f'{round_name.lower().replace(" ", "_")}_accuracies.png'))
    plt.close()
    
    print(f"\nBest {round_name} model: {type(best_model).__name__}")
    print(f"Best accuracy: {best_score:.4f}")
    return best_model

# Evaluate early round models (using only seed features)
early_features = X_train_scaled[['Team1_SeedNum', 'Team2_SeedNum']]
early_features_test = X_test_scaled[['Team1_SeedNum', 'Team2_SeedNum']]
early_model = evaluate_models(early_models, early_features, early_features_test, 
                            y_train, y_test, "Early Round")

# Evaluate middle round models (using all features)
middle_model = evaluate_models(middle_models, X_train_scaled, X_test_scaled, 
                              y_train, y_test, "Middle Round")

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
                'Team2_AvgPoints': team2_feat['AvgPoints'],
                'Team2_AvgOppPoints': team2_feat['AvgOppPoints'],
                'Team2_SeedNum': team2_feat['SeedNum']
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
        
        # Return winner based on probability
        return team1_id if model_probs[0][1] > 0.5 else team2_id
    
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
        season_seeds = tourney_seeds[tourney_seeds['Season'] == self.season]
        teams = season_seeds['TeamID'].tolist()
        
        # Create initial matchups based on seeding
        matchups = []
        n = len(teams)
        for i in range(0, n, 2):
            if i + 1 < n:
                team1, team2 = teams[i], teams[i+1]
                matchups.append((team1, team2))
        
        # Run simulations for each matchup
        for team1, team2 in matchups:
            if (team1, team2) not in self.matchup_stats and (team2, team1) not in self.matchup_stats:
                self.matchup_stats[(team1, team2)] = {team1: 0, team2: 0}
                
                # Simulate this matchup multiple times
                for _ in range(self.num_simulations):
                    bracket = Bracket(self.season, self.model, self.scaler, current_round=self.current_round)
                    winner = bracket.simulate_matchup(team1, team2)
                    if winner:
                        self.matchup_stats[(team1, team2)][winner] += 1
                
                self.brackets.append(bracket)
                
                # Update round number for next set of matchups
                if len(self.matchup_stats) % 32 == 0:  # First round complete
                    self.current_round = 2
                elif len(self.matchup_stats) % 16 == 0:  # Second round complete
                    self.current_round = 3
                elif len(self.matchup_stats) % 8 == 0:  # Sweet 16 complete
                    self.current_round = 4
                elif len(self.matchup_stats) % 4 == 0:  # Elite 8 complete
                    self.current_round = 5
                elif len(self.matchup_stats) % 2 == 0:  # Final Four complete
                    self.current_round = 6
    
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
        # Split matchups into groups of 8 for better visualization
        matchup_items = list(self.matchup_stats.items())
        num_groups = (len(matchup_items) + 7) // 8
        
        for group in range(num_groups):
            start_idx = group * 8
            end_idx = min(start_idx + 8, len(matchup_items))
            current_matchups = matchup_items[start_idx:end_idx]
            
            plt.figure(figsize=(12, 10))
            
            for i, (matchup, stats) in enumerate(current_matchups):
                team1, team2 = matchup
                team1_name = self.brackets[0].get_team_name(team1)
                team2_name = self.brackets[0].get_team_name(team2)
                
                total = sum(stats.values())
                team1_pct = (stats[team1] / total) * 100
                team2_pct = (stats[team2] / total) * 100
                
                # Create a subplot for each matchup
                ax = plt.subplot(len(current_matchups), 1, i+1)
                
                # Create the horizontal bars
                bars1 = plt.barh([0], [team1_pct], label=team1_name, color='#3498db', height=0.5)
                bars2 = plt.barh([0], [team2_pct], left=[team1_pct], label=team2_name, color='#e74c3c', height=0.5)
                
                # Add percentage labels
                if team1_pct > 5:  # Only show label if bar is wide enough
                    plt.text(team1_pct/2, 0, f'{team1_pct:.1f}%', 
                            ha='center', va='center', color='white', fontweight='bold')
                if team2_pct > 5:
                    plt.text(team1_pct + team2_pct/2, 0, f'{team2_pct:.1f}%', 
                            ha='center', va='center', color='white', fontweight='bold')
                
                # Customize the subplot
                plt.yticks([])
                plt.xlim(0, 100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Add a title that shows seeds if available
                team1_seed = tourney_seeds[
                    (tourney_seeds['Season'] == self.season) & 
                    (tourney_seeds['TeamID'] == team1)
                ]['SeedNum'].iloc[0] if not tourney_seeds.empty else '?'
                
                team2_seed = tourney_seeds[
                    (tourney_seeds['Season'] == self.season) & 
                    (tourney_seeds['TeamID'] == team2)
                ]['SeedNum'].iloc[0] if not tourney_seeds.empty else '?'
                
                plt.title(f'({team1_seed}) {team1_name} vs ({team2_seed}) {team2_name}', 
                          loc='left', pad=5, fontsize=10)
                
                # Add legend
                if i == 0:  # Only show legend for the first subplot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'Tournament Matchup Predictions - Group {group + 1}', 
                         fontsize=14, y=0.95)
            plt.tight_layout()
            plt.savefig(f'matchup_predictions_group_{group+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_tournament_bracket(self):
        plt.figure(figsize=(20, 12))
        
        # Define the number of rounds and teams per round
        rounds = ['First Round', 'Second Round', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
        teams_per_round = [32, 16, 8, 4, 2, 1]
        
        # Calculate positions for each team in each round
        spacing = 1
        round_positions = {}
        for round_idx, num_teams in enumerate(teams_per_round):
            y_positions = np.linspace(0, len(teams_per_round) * spacing, num_teams)
            round_positions[round_idx] = y_positions
        
        # Draw connecting lines and team names
        matchup_idx = 0
        matchup_items = list(self.matchup_stats.items())
        
        for round_idx, num_teams in enumerate(teams_per_round[:-1]):
            x_start = round_idx * 3  # Horizontal spacing between rounds
            x_end = x_start + 3
            
            for i in range(0, num_teams, 2):
                if matchup_idx >= len(matchup_items):
                    break
                    
                y1 = round_positions[round_idx][i]
                y2 = round_positions[round_idx][i + 1]
                y_next = round_positions[round_idx + 1][i // 2]
                
                # Get teams for this matchup
                matchup, stats = matchup_items[matchup_idx]
                team1, team2 = matchup
                team1_name = self.brackets[0].get_team_name(team1)
                team2_name = self.brackets[0].get_team_name(team2)
                
                # Get win percentages
                total = sum(stats.values())
                team1_pct = (stats[team1] / total) * 100
                team2_pct = (stats[team2] / total) * 100
                
                # Draw lines with thickness based on win probability
                plt.plot([x_start, x_end], [y1, y_next], 'b-', 
                         alpha=team1_pct/100, linewidth=2*team1_pct/100)
                plt.plot([x_start, x_end], [y2, y_next], 'r-', 
                         alpha=team2_pct/100, linewidth=2*team2_pct/100)
                
                # Add team names and win percentages
                plt.text(x_start - 0.2, y1, f'{team1_name}\n({team1_pct:.1f}%)', 
                         ha='right', va='center', fontsize=8)
                plt.text(x_start - 0.2, y2, f'{team2_name}\n({team2_pct:.1f}%)', 
                         ha='right', va='center', fontsize=8)
                
                matchup_idx += 1
        
        # Add round labels
        for round_idx, round_name in enumerate(rounds):
            plt.text(round_idx * 3, len(teams_per_round) * spacing + 0.5, 
                     round_name, ha='center', va='bottom', fontsize=12)
        
        # Customize the plot
        plt.title('NCAA Tournament Bracket Simulation Results', fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('tournament_bracket.png', dpi=300, bbox_inches='tight')
        plt.close()

class TournamentPredictor:
    def __init__(self, season, early_model=None, middle_model=None, final_model=None):
        self.season = season
        # Models are passed in after PyCaret selection
        self.early_model = early_model
        self.middle_model = middle_model
        self.final_model = final_model
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train, X_test, y_test, round_info):
        # Extract seed features for early round model
        early_features = X_train[['Team1_SeedNum', 'Team2_SeedNum']]
        early_features_test = X_test[['Team1_SeedNum', 'Team2_SeedNum']]
        
        # Evaluate early round model
        early_preds = self.early_model.predict(early_features_test)
        early_score = accuracy_score(y_test, early_preds)
        print(f"Early Round Model Score: {early_score}")
        print(classification_report(y_test, early_preds))
        
        # Evaluate middle round model
        middle_preds = self.middle_model.predict(X_test)
        middle_score = accuracy_score(y_test, middle_preds)
        print(f"Middle Round Model Score: {middle_score}")
        print(classification_report(y_test, middle_preds))
        
        # Evaluate final round model
        final_preds = self.final_model.predict(X_test)
        final_score = accuracy_score(y_test, final_preds)
        print(f"Final Round Model Score: {final_score}")
        print(classification_report(y_test, final_preds))
    
    def predict_proba(self, features, round_num):
        if round_num <= 2:
            # Early rounds: use seed-based model
            seed_features = features[['Team1_SeedNum', 'Team2_SeedNum']]
            return {'early': self.early_model.predict_proba(seed_features)}
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
tourney_rounds['Round'] = (tourney_rounds['DayNum'] - 134) // 2 + 1  # Convert DayNum to tournament round

# Create and train the tournament predictor with PyCaret models
season_to_simulate = 2024  # Current season
predictor = TournamentPredictor(season_to_simulate, early_model, middle_model, final_model)
predictor.train(X_train_scaled, y_train, X_test_scaled, y_test, tourney_rounds)

# Run tournament simulation with round-based predictions
print("\nTournament Predictions (Round-Based):")
simulation = BracketSimulation(season_to_simulate, predictor, scaler, num_simulations=100)
simulation.simulate_brackets()
simulation.visualize_matchup_stats()

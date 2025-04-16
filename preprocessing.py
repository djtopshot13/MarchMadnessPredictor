# # Importing Libraries  
# from functools import reduce
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe 
# from IPython.display import display, HTML  
# import numpy as np 
# import pandas as pd 
# from sklearn.metrics import accuracy_score 
# from sklearn.model_selection import train_test_split  
# from sklearn.preprocessing import MinMaxScaler  
# import warnings 
# from xgboost import XGBRegressor  

# pd.set_option('display.max_columns', None)  
# warnings.filterwarnings('ignore')  

# HTML("""
# <style>
# g.pointtext {display: none;}
# </style>
# """)

import pandas as pd
import re
import os

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

def load_data(file_name):
    return pd.read_csv(os.path.join(project_root, 'MarchMadnessData', file_name))

conf_df = load_data("Conferences.csv")
cities_df = load_data("Cities.csv")
teams_df = load_data("MTeams.csv")

def write_data(df, file_name):
    filepath = os.path.join(project_root, 'MarchMadnessData', file_name)
    df.to_csv(filepath, index=False)

def clean_tourney_games():
    file_name = "MConferenceTourneyGames.csv"
    tg_df = load_data(file_name)

    tg_df.drop(columns=["LName", "WName", "ConfName"], inplace=True)

    
    tg_df.insert(column="ConfName", loc=2, value=None)
    tg_df.insert(column="WName", loc=4, value=None)
    tg_df.insert(column="LName", loc=6, value=None)
    

    # Create dictionaries for to populate team and conference names properly to their respective names
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))
    conf_dict = dict(zip(conf_df.ConfAbbrev, conf_df.Description))


    tg_df["WName"] = tg_df["WTeamID"].map(team_dict)
    tg_df["LName"] = tg_df["LTeamID"].map(team_dict)    
    tg_df["ConfName"] = tg_df["ConfAbbrev"].map(conf_dict)
    
    write_data(tg_df, file_name)

def clean_game_cities():
    """Day number range for regular season games is 7-132
    After that is NCAA or Secondary Tournaments which is 133-154
    """

    file_name = "MGameCities.csv"
    gc_df = load_data(file_name)

    gc_df.drop(columns=["LName", "WName"], inplace=True)

    gc_df.insert(column="WName", loc=2, value=None)
    gc_df.insert(column="LName", loc=4, value=None)

    # Create dictionaries for to populate team and conference names properly to their respective names
    city_dict = dict(zip(cities_df.CityID, cities_df.City))
    state_dict = dict(zip(cities_df.CityID, cities_df.State))
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    gc_df["CityName"] = gc_df["CityID"].map(city_dict)
    gc_df["State"] = gc_df["CityID"].map(state_dict)
    gc_df["WName"] = gc_df["WTeamID"].map(team_dict)
    gc_df["LName"] = gc_df["LTeamID"].map(team_dict)
    
    write_data(gc_df, file_name)

def clean_ncaa_tourney_compact_results():
    file_name = "MNCAATourneyCompactResults.csv"
    ntcr_df = load_data(file_name)

    ntcr_df.drop(columns=["LName", "WName"], inplace=True)

    ntcr_df.insert(column="WName", loc=2, value=None)
    ntcr_df.insert(column="LName", loc=5, value=None)

    # Create dictionaries for to populate team names properly to their respective names
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    ntcr_df["WName"] = ntcr_df["WTeamID"].map(team_dict)
    ntcr_df["LName"] = ntcr_df["LTeamID"].map(team_dict)

    write_data(ntcr_df, file_name)

def clean_ncaa_tourney_detailed_results():

    file_name = "MNCAATourneyDetailedResults.csv"
    ntdr_df = load_data(file_name)

    ntdr_df.drop(columns=["LName", "WName"], inplace=True)

    ntdr_df.insert(column="WName", loc=2, value=None)
    ntdr_df.insert(column="LName", loc=5, value=None)

    # Create dictionaries for to populate team names properly to their respective names
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    ntdr_df["WName"] = ntdr_df["WTeamID"].map(team_dict)
    ntdr_df["LName"] = ntdr_df["LTeamID"].map(team_dict)

    write_data(ntdr_df, file_name)

def clean_tourney_seeds():
    file_name = "MNCAATourneySeeds.csv"

    ts_df = load_data(file_name)
    ts_df.drop(columns=["TeamName"], inplace=True)

    ts_df.insert(column="TeamName", loc=2, value=None)

    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))
    ts_df["TeamName"] = ts_df["TeamID"].map(team_dict)

    write_data(ts_df, file_name)

"""
Look at getting the tournament values populated past the first round possibly 
for modeling the tournament results more easily. Use the NCAATourneyCompactResults.csv
to find the winning teams and get the proper round mapping for the winning team to be used
"""
def region_order(slot):
    """Return region precedence: W=0, X=1, Y=2, Z=3, else 99."""
    for i, region in enumerate(['W', 'X', 'Y', 'Z']):
        if region in slot:
            return i
    return 99

def pick_strong_weak(seed1, seed2, slot1, slot2):
    """Return (strong_seed, weak_seed, strong_slot, weak_slot) with tiebreaker."""
    num1, num2 = int(seed1[1:]), int(seed2[1:])
    if num1 < num2:
        return seed1, seed2, slot1, slot2
    elif num2 < num1:
        return seed2, seed1, slot2, slot1
    else:
        # Seeds are equal, use region order
        if region_order(slot1) < region_order(slot2):
            return seed1, seed2, slot1, slot2
        else:
            return seed2, seed1, slot2, slot1

def clean_tourney_slots():
    file_name = "MNCAATourneySlots.csv"
    tsl_df = load_data(file_name)
    tsl_df.drop(columns=["StrongTeamName", "WeakTeamName"], inplace=True)

    ts_df = load_data("MNCAATourneySeeds.csv")
    tcr_df = load_data("MNCAATourneyCompactResults.csv")

    # Build mapping dictionaries with (Season, ...) keys for multi-season support
    seed_to_team = ts_df.set_index(['Season', 'Seed'])['TeamName'].to_dict()
    team_to_seed = ts_df.set_index(['Season', 'TeamName'])['Seed'].to_dict()
    teamid_to_team = ts_df.set_index(['Season', 'TeamID'])['TeamName'].to_dict()
    team_to_teamid = ts_df.set_index(['Season', 'TeamName'])['TeamID'].to_dict()

    # Fill in first round team names using seeds
    tsl_df['StrongTeamName'] = tsl_df.apply(
        lambda row: seed_to_team.get((row['Season'], row['StrongSeed'])), axis=1)
    tsl_df['WeakTeamName'] = tsl_df.apply(
        lambda row: seed_to_team.get((row['Season'], row['WeakSeed'])), axis=1)

    # Build winner lookup: (Season, TeamID1, TeamID2) -> Winner TeamID
    game_winners = {}
    for _, row in tcr_df.iterrows():
        season = row['Season']
        wtid, ltid = row['WTeamID'], row['LTeamID']
        game_winners[(season, wtid, ltid)] = wtid
        game_winners[(season, ltid, wtid)] = wtid

    slots_filled = tsl_df.copy()
    slot_to_winner = {}  # (Season, Slot) -> Winner TeamName

    all_seasons = tsl_df['Season'].unique()

    for season in all_seasons:
        if season == 2020:
            continue
        # First round: use original team names, determine winners
        r1_mask = (slots_filled['Season'] == season) & (slots_filled['Slot'].str.startswith('R1'))
        for idx, row in slots_filled[r1_mask].iterrows():
            strong_team = row['StrongTeamName']
            weak_team = row['WeakTeamName']
            strong_id = team_to_teamid.get((season, strong_team))
            weak_id = team_to_teamid.get((season, weak_team))
            winner_id = game_winners.get((season, strong_id, weak_id))
            winner_name = teamid_to_team.get((season, winner_id))
            slot_to_winner[(season, row['Slot'])] = winner_name

        # Subsequent rounds
        for round_num in range(2, 7):  # R2 to R6
            round_prefix = f'R{round_num}'
            r_mask = (slots_filled['Season'] == season) & (slots_filled['Slot'].str.startswith(round_prefix))
            for idx, row in slots_filled[r_mask].iterrows():
                strong_prev = row['StrongSeed']
                weak_prev = row['WeakSeed']
                strong_team = slot_to_winner.get((season, strong_prev))
                weak_team = slot_to_winner.get((season, weak_prev))
                # Get seeds for tiebreaker
                strong_seed = team_to_seed.get((season, strong_team))
                weak_seed = team_to_seed.get((season, weak_team))
                # Use slot names for tiebreaker
                strong_slot = strong_prev
                weak_slot = weak_prev
                # Pick strong/weak using tiebreaker
                if strong_seed and weak_seed:
                    s_seed, w_seed, s_slot, w_slot = pick_strong_weak(strong_seed, weak_seed, strong_slot, weak_slot)
                    s_team = slot_to_winner.get((season, s_slot))
                    w_team = slot_to_winner.get((season, w_slot))
                    slots_filled.at[idx, 'StrongTeamName'] = s_team
                    slots_filled.at[idx, 'WeakTeamName'] = w_team
                    # Get winner
                    if s_team and w_team:
                        s_id = team_to_teamid.get((season, s_team))
                        w_id = team_to_teamid.get((season, w_team))
                        winner_id = game_winners.get((season, s_id, w_id))
                        winner_name = teamid_to_team.get((season, winner_id))
                        slot_to_winner[(season, row['Slot'])] = winner_name

    write_data(slots_filled, file_name)


def clean_regular_season_compact_results():
    file_name = "MRegularSeasonCompactResults.csv"
    rsc_df = load_data(file_name)

    rsc_df.drop(columns=["LName", "WName"], inplace=True)

    rsc_df.insert(column="WName", loc=2, value=None)
    rsc_df.insert(column="LName", loc=5, value=None)

    # Create dictionaries for to populate team names properly to their respective names
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    rsc_df["WName"] = rsc_df["WTeamID"].map(team_dict)
    rsc_df["LName"] = rsc_df["LTeamID"].map(team_dict)

    write_data(rsc_df, file_name)

def clean_regular_season_detailed_results():

    file_name = "MRegularSeasonDetailedResults.csv"
    rsd_df = load_data(file_name)

    rsd_df.drop(columns=["LName", "WName"], inplace=True)

    rsd_df.insert(column="WName", loc=2, value=None)
    rsd_df.insert(column="LName", loc=5, value=None)

    # Create dictionaries for to populate team names properly to their respective names
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    rsd_df["WName"] = rsd_df["WTeamID"].map(team_dict)
    rsd_df["LName"] = rsd_df["LTeamID"].map(team_dict)

    write_data(rsd_df, file_name)

def clean_seasons():
    file_name = "MSeasons.csv"
    s_df = load_data(file_name)

    s_df.drop(columns=["SeasonStart","SeasonEnd"], inplace=True)

    s_df.insert(column="SeasonStart", loc=1, value=None)
    s_df.insert(column="SeasonEnd", loc=2, value=None)

    s_df["SeasonStart"] = pd.to_datetime(s_df["DayZero"])
    s_df["SeasonEnd"] = pd.to_datetime(s_df["DayZero"]) + pd.to_timedelta(154, unit='D')

    s_df.drop(columns=["DayZero"], inplace=True)
    
    write_data(s_df, file_name)


def capitalize_name(name):
    # Split the name into parts
    parts = re.split(r'[_\s]+', name)
    
    # Capitalize each part, with special handling for 'mc' and apostrophes
    capitalized_parts = []
    for part in parts:
        # Handle 'mc' prefix
        if part.lower().startswith('mc'):
            part = 'Mc' + part[2].upper() + part[3:]
        else:
            # Split by apostrophe and capitalize each subpart
            subparts = part.split("'")
            part = "'".join(subpart.capitalize() for subpart in subparts)
        
        capitalized_parts.append(part)
    
    return ' '.join(capitalized_parts)

def add_period_to_st(input_file, output_file):
    # Load the data
    df = load_data(input_file)

    # Check if the column containing college names is named 'CollegeName'
    if 'TeamName' in df.columns:
        # Use regex to add a period after 'St' at the end of names
        df['TeamName'] = df['TeamName'].str.replace(r'St\b ', 'St. ', regex=True)

    write_data(df, output_file)


def clean_team_coaches():
    file_name = "MTeamCoaches.csv"
    tc_df = load_data(file_name)

    tc_df.drop(columns=["TeamName"], inplace=True)

    tc_df.insert(column="TeamName", loc=1, value=None)

    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    tc_df["TeamName"] = tc_df["TeamID"].map(team_dict)

    tc_df["CoachName"] = tc_df["CoachName"].apply(capitalize_name)

    write_data(tc_df, file_name)

def clean_team_conferences():
    file_name = "MTeamConferences.csv"
    t_conf_df = load_data(file_name)
    
    t_conf_df.drop(columns=["TeamName", "ConfName"], inplace=True)

    t_conf_df.insert(column="TeamName", loc=1, value=None)
    t_conf_df.insert(column="ConfName", loc=4, value=None)

    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))
    conf_dict = dict(zip(conf_df.ConfAbbrev, conf_df.Description))

    t_conf_df["TeamName"] = t_conf_df["TeamID"].map(team_dict)
    t_conf_df["ConfName"] = t_conf_df["ConfAbbrev"].map(conf_dict)
    
    write_data(t_conf_df, file_name)


# add_period_to_st("MarchMadnessData/MTeams.csv", "MarchMadnessData/MTeams.csv")
# clean_tourney_games()
# clean_game_cities()
# clean_ncaa_tourney_compact_results()
# clean_ncaa_tourney_detailed_results()

# clean_tourney_seeds()

clean_tourney_slots()
# # ^^^This function needs some more proofing


# clean_regular_season_compact_results()
# clean_regular_season_detailed_results()
# # clean_seasons()
# clean_team_coaches()
# clean_team_conferences()

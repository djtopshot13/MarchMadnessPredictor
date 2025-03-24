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

def load_data(file_path):
    return pd.read_csv(file_path)

conf_df = load_data("MarchMadnessData/Conferences.csv")
cities_df = load_data("MarchMadnessData/Cities.csv")
teams_df = load_data("MarchMadnessData/MTeams.csv")
cities_df = load_data("MarchMadnessData/Cities.csv")

def write_data(df, filepath):
    df.to_csv(filepath, index=False)

def clean_tourney_games():
    filepath = "MarchMadnessData/MConferenceTourneyGames.csv"
    tg_df = load_data(filepath)

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
    
    write_data(tg_df, filepath)

def clean_game_cities():
    """Day number range for regular season games is 7-132
    After that is NCAA or Secondary Tournaments which is 133-154
    """

    filepath = "MarchMadnessData/MGameCities.csv"
    gc_df = load_data(filepath)

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
    
    write_data(gc_df, filepath)

def clean_ncaa_tourney_compact_results():
    filepath = "MarchMadnessData/MNCAATourneyCompactResults.csv"
    ntcr_df = load_data(filepath)

    ntcr_df.drop(columns=["LName", "WName"], inplace=True)

    ntcr_df.insert(column="WName", loc=2, value=None)
    ntcr_df.insert(column="LName", loc=5, value=None)

    # Create dictionaries for to populate team names properly to their respective names
    team_dict = dict(zip(teams_df.TeamID, teams_df.TeamName))

    ntcr_df["WName"] = ntcr_df["WTeamID"].map(team_dict)
    ntcr_df["LName"] = ntcr_df["LTeamID"].map(team_dict)


    
    write_data(ntcr_df, filepath)

# clean_tourney_games()
# clean_game_cities()
clean_ncaa_tourney_compact_results()

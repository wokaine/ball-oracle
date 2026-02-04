from understatapi import UnderstatClient
import pandas as pd

def enum_position(pos):
    match pos:
        case "F":
            # Forward
            return 0
        case "S":
            # Striker
            return 1
        case "F S":
            return 2
        case "M":
            # Midfielder
            return 3
        case "F M":
            return 4
        case "D":
            # Defender
            return 5
        case "D F M":
            return 6
        case "F M S":
            return 7
        case "M S":
            return 8
        case "D F M S":
            return 9
        case "D M S":
            return 10
        case "D S":
            return 11
        case "D M":
            return 12
        case "D F S":
            return 13
        case "GK":
            return 14
        case "GK S":
            return 15
        case "GK M":
            return 16
        case "GK F":
            return 17
        case "GK D":
            return 18
        case _:
            raise Exception(f"ERROR: no enum found for {pos}, consider adding this to the function")

def fetch_understat_players(leagues=["EPL"], season="2025"):
    df_list = []

    with UnderstatClient() as understat:
        for l in leagues:
            league_players = understat.league(league=l).get_player_data(season=season)
            df = pd.DataFrame(league_players)
            numeric_cols = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes', 'yellow_cards', 'red_cards', 'xGChain', 'xGBuildup', 'time']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df
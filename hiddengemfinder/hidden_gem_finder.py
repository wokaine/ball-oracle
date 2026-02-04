import sys
from sklearn.cluster import KMeans
from pathlib import Path

from textual import log

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from lib import utils

class HiddenGemFinder():
    def __init__(self, leagues=None, player=None):
        self.leagues = leagues
        self.player = player
        self.query = None
        self.data = None
        if leagues is not None:
            self.setup()
        if player is not None:
            self.find_similar_players(player)
    
    def get_query(self):
        return self.query
    
    def update_league(self, leagues):
        log("UPDATE CALLED")
        # Idempotency check
        if self.leagues != leagues:
            log(f"IDEMPOTENCY CHECK SUCCESS. self: {self.leagues}, leagues: {leagues}")
            self.leagues = leagues
            self.setup()

    def update_player(self, player):
        # Idempotency check
        if self.player != player:
            self.player = player
            self.find_similar_players()

    def setup(self):
        log("SETUP CALLED")
        df = self.calculate_features(utils.fetch_understat_players(leagues=self.leagues))
        km = KMeans(n_clusters=10, max_iter=500, random_state=42)
        # Definitely do not cluster based on id, player name or team name
        # Games played and time should also be unimportant
        df_features = df.drop(['id', 'player_name', 'games', 'time', '90s', 'team_title'], axis=1)
        df_features['position'] = df_features['position'].apply(utils.enum_position)
        df['cluster_label'] = km.fit_predict(df_features)
        self.data = df

    def calculate_features(self, df):
        # Calculate some more features for clustering
        df['90s'] = df['time'] / 90
        df['xG_per_shot'] = df['xG'] / df['shots']
        df['key_passes_per_90'] = df['key_passes'] / df['90s']
        df['xGBuildup_ratio'] = df['xGBuildup'] / df['xGChain']
        df['xGChain_per_90'] = df['xGChain'] / df['90s']
        df = df.fillna(0)

        # Only return players who have played more than 450 mins (5 full games)
        df = df[df['time'] >= 450].copy()
        return df
    
    def find_similar_players(self):
        # Return just a subset of the full table, no extra queries needed

        df = self.data

        if self.player not in df['player_name'].values:
            return None
        
        cluster = df.loc[df['player_name'].str.lower() == self.player.lower(), 'cluster_label'].values[0]
        self.query = df[df['cluster_label'] == cluster]


    
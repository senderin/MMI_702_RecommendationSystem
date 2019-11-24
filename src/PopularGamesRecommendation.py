from src.Data import Data


class PopularGamesRecommendation():

    gameRating = None
    played_games = None

    def __init__(self):
        dfs = Data.get_instance()
        self.gameRating = dfs.game_ratings
        self.played_games = dfs.played_games.copy()

    def get_most_played_games(self):
        # most-played games
        most_played_10 = self.gameRating.sort_values('average_hours', ascending=False).head(10)
        return most_played_10

    def get_most_rated_games(self):
        # most-rated games
        most_rated_10 = self.gameRating.sort_values('rating', ascending=False).head(10)
        return most_rated_10

    def get_most_popular_games(self):
        # popularity_recs = users_games.Game_ID.value_counts().head(10).index.tolist()
        # most-popular games
        most_popular_10 = self.gameRating.sort_values('number_of_player', ascending=False).head(10)
        return most_popular_10

    def recommend_for_user_set(self, test):
        # make recommendations for all members in the test data
        popularity_recs = self.get_most_popular_games()['Game_ID'].values.tolist()
        recommendations = []
        for user in test.index:
            recommendations.append(popularity_recs)
        return recommendations

    def recommend_for_user(self):
        ids = self.get_most_popular_games()['Game_ID'].values.tolist()
        names = self.get_most_popular_games()['Game_Name'].values.tolist()
        return ids, names

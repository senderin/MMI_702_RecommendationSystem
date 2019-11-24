from src.Data import Data


class RandomGamesRecommendation():

    users_games = None
    played_games = None

    def __init__(self):
        dfs = Data.get_instance()
        self.users_games = dfs.users_games.copy()
        self.played_games = dfs.played_games.copy()

    def recommend_for_user_set(self, test):
        recommendations = []
        for user in test.index:
            random_predictions = self.played_games.Game_ID.sample(10).values.tolist()
            recommendations.append(random_predictions)
        return recommendations

    def recommend_for_user(self):
        ids = self.played_games.Game_ID.sample(10).values.tolist()
        names = self.played_games[self.played_games['Game_ID'].isin(ids)]['Game_Name'].values.tolist()
        return ids, names
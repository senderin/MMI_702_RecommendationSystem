from src.Dataframes import Dataframes


class RandomGamesRecommendation():

    users_games = None

    def __init__(self):
        dfs = Dataframes.get_instance()
        self.users_games = dfs.users_games

    def recommend_for_user_set(self, test):
        recommendations = []
        for user in test.index:
            random_predictions = self.users_games.Game_ID.sample(10).values.tolist()
            recommendations.append(random_predictions)
        return recommendations
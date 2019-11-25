from src.Data import Data
import pandas as pd


class SimilarGamesRecommendation():

    users_games = None
    game_id_name = None

    def __init__(self):
        dfs = Data.get_instance()
        self.users_games = dfs.users_games.copy()
        self.game_id_name = dfs.game_id_name.copy()

    def get_similar_games(self, game_name):
        # user - game matrix
        user_game_matrix = self.users_games.pivot_table(index='User_ID', columns='Game_Name', values='Hours').fillna(0)
        # print(user_game_matrix.head(3))
        game_X = user_game_matrix[game_name]
        similar_to_X = user_game_matrix.corrwith(game_X)
        corr_contact = pd.DataFrame(similar_to_X, columns=['Correlation'])
        # drop those null values
        corr_contact.dropna(inplace=True)
        corr_contact = corr_contact.sort_values('Correlation', ascending=False).head(11)
        print(corr_contact)
        return corr_contact

    def recommend_for_game(self, game_name):
        game_name = str.lower(game_name)
        similarity_recs = self.get_similar_games(game_name).merge(self.game_id_name, on='Game_Name').head(11)
        names = similarity_recs['Game_Name'].values.tolist()
        names.pop(0)
        return names

    def recommend_for_user_set(self, test, game_name):
        game_name = str.lower(game_name)
        similarity_recs = self.get_similar_games(game_name).merge(self.game_id_name, on='Game_Name').head(11)
        recommendations = []
        for user in test.index:
            first_rec = similarity_recs['Game_ID'].values.tolist()
            recommendations.append(first_rec)
        return recommendations
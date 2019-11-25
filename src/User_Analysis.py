from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from src.Data import Data

class User_Analysis():

    data = None
    playtime = None
    game_id_name = None
    steam_app_data = None
    users = None
    user_id = 0
    user_info = None
    game_ids = []
    games_info = []

    def __init__(self, user_id):
        self.data = Data.get_instance()
        self.playtime = self.data.playtime
        self.game_id_name = self.data.game_id_name
        self.steam_app_data = self.data.steam_app_data
        self.users = self.create_users_df()
        self.user_id = user_id
        self.user_info = pd.DataFrame(self.data.users_games.loc[self.data.users_games['User_ID'] == self.user_id])
        self.game_ids = self.user_info['Game_ID'].tolist()
        self.games_info = self.data.played_games.loc[self.data.played_games['Game_ID'].isin(self.game_ids)]

    def get_actual_id_list(self):
        grouped = self.users.groupby('User_ID')
        #print(grouped.head(10))
        ids = grouped.get_group(self.user_id)['Game_ID'].values.tolist()
        return ids

    def get_actual_name_list(self):
        ids = self.get_actual_id_list()
        names = self.data.played_games.loc[self.data.played_games['Game_ID'].isin(ids)]['Game_Name'].values.tolist()
        return names

    def create_users_df(self):
        temp = self.data.users_games.copy()
        temp = temp.groupby('User_ID').filter(lambda x: len(x) > 10)
        temp.drop("Hours", inplace=True, axis=1)
        temp.drop("Game_Name", inplace=True, axis=1)
        return temp

    def user_profile_description(self):
        cat_list =  self.games_info['categories'].tolist()
        genre_list = self.games_info['genres'].tolist()

        most_played_3_games = self.get_most_played_N_games(10)

        genre_value_counts, cat_value_counts = self.plot_genre_cat_preferences(genre_list, cat_list)

        user_ratings = pd.merge(self.user_info, self.data.game_ratings, on='Game_ID')
        user_ratings['user_rating'] = user_ratings['Hours']/user_ratings['average_hours']
        user_ratings['user_rating_2'] = user_ratings['Hours']/user_ratings['Hours'].mean()
        # user_rating ve user_rating_2 paralel
        user_ratings['user_rating_3'] = user_ratings['user_rating']*user_ratings['user_rating_2']
        user_ratings.sort_values(by=['user_rating'])
        #sns.jointplot(x='user_rating', y='user_rating_3', data=user_info)
        #plt.show()

        print('How many games have the player played in total?')
        print('{0}'.format(len(self.game_ids)))
        print('How many hours have players played in the game?')
        print('{0}'.format(sum(user_ratings['Hours'])))
        print('What is the average playing time of the player?')
        print('{0}'.format(sum(user_ratings['Hours']) / len(self.game_ids)))
        print('What are the three most played games by the player?')
        print(most_played_3_games['Game_Name'])
        print('What genre of games does the player play the most?')
        print(genre_value_counts[:10])
        print('What category of tag games did the player play the most?')
        print(cat_value_counts[:10])

    def plot_word_cloud(self, data):
        wordcloud = WordCloud(background_color="white", stopwords=set(STOPWORDS))
        word_could_dict = Counter(data)
        wordcloud.generate_from_frequencies(word_could_dict)
        plt.figure(figsize=(5, 3))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def get_most_played_N_games(self, N):
        games = self.user_info.sort_values(by=['Hours']).tail(N)
        return self.data.played_games.loc[self.data.played_games['Game_ID'].isin(games['Game_ID'])]

    def plot_genre_cat_preferences(self, genre_list, cat_list):
        genres = str(genre_list).replace("'", '').replace('[', '').replace(']', '').replace(',', ';').replace(' ', '').split(';')
        # for genre
        #self.plot_word_cloud(genres)

        categories = str(cat_list).replace("'", '').replace('[', '').replace(']', '').replace(',', ';').replace(' ', '').split(';')
        # for categories
        #self.plot_word_cloud(categories)

        #pd.Series(genres).value_counts().plot(kind='bar')
        #pd.Series(categories).value_counts().plot(kind='bar')

        return  pd.Series(genres).value_counts(), pd.Series(categories).value_counts()










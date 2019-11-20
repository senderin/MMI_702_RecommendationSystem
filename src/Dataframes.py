import string
import seaborn as sns
import pandas as pd
import recmetrics
import matplotlib.pyplot as plt
from ast import literal_eval

from pandas import array


class Dataframes():

    __instance = None
    playtime = pd.DataFrame()
    game_id_name = pd.DataFrame()
    users_games = pd.DataFrame()
    game_ratings = pd.DataFrame()
    games_info_10000 = pd.DataFrame()
    played_games_info = pd.DataFrame()

    @staticmethod
    def get_instance():
        if Dataframes.__instance == None:
            Dataframes()
        return Dataframes.__instance

    def __init__(self):
        if Dataframes.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Dataframes.__instance = self

        self.read_csvs()


    def read_csvs(self):
        # reading datasets
        self.game_id_name = pd.read_csv('game_id_name.csv', sep='\t')
        self.games_info_10000 = pd.read_csv('steam_app_data.csv')

        self.preprocessing_on_game_infos()
        self.process_categories_and_genres()

        self.playtime = pd.read_csv('playtime.csv', sep='\t')
        # eliminate ones that have no corresponding name
        playtime = self.playtime.loc[self.playtime['Game_ID'].isin(self.game_id_name['Game_ID'])]

        self.users_games = pd.merge(self.game_id_name, self.playtime, on='Game_ID')
        user_ids = self.users_games['User_ID'].unique()
        print("# of users: {0}".format(len(user_ids)))
        game_ids = self.users_games['Game_ID'].unique()
        print("# of games: {0}".format(len(game_ids)))


        self.game_ratings = self.get_game_ratings()

        print()
        print("played_game_infos {0}".format(len(self.played_games_info['Game_ID'].unique())))
        print("game_id_name {0}".format(len(self.game_id_name['Game_ID'].unique())))
        print("playtime {0}".format(len(self.playtime['Game_ID'].unique())))
        print("game_ratings {0}".format(len(self.game_ratings['Game_ID'].unique())))
        print("users_games {0}".format(len(game_ids)))

    def get_game_ratings(self):
        # calculate the average playtime for each game
        mean = self.users_games['Hours'].mean()
        gameRating = pd.DataFrame(self.game_id_name)
        gameRating['total_hours'] = pd.DataFrame(self.users_games.groupby(['Game_ID'])['Hours'].sum())
        gameRating['number_of_player'] = self.users_games.groupby(['Game_ID'])['Game_ID'].count()
        gameRating['average_hours'] = pd.DataFrame(self.users_games.groupby(['Game_ID'])['Hours'].mean())
        # calculate ratings
        gameRating['rating'] = gameRating['average_hours'] / mean
        # eliminate games played less than 1% of players
        threshold = len(self.users_games['User_ID'].unique())*1/100
        # print('threshold: {0}'.format(threshold))
        gameRating = gameRating.loc[gameRating['number_of_player'] > threshold]

        # number of players
        v = gameRating['number_of_player']
        #  the minimum players required
        m = gameRating['number_of_player'].quantile(0.90)
        # average hour
        R = gameRating['average_hours']
        # the mean hours across the whole report
        C = gameRating['average_hours'].mean()

        gameRating['weighted_rating'] = (v/(v+m) * R) + (m/(m+v) * C)

        return gameRating

    def preprocessing_on_game_infos(self):

        # dropping ALL duplicate values
        self.game_id_name.drop_duplicates(subset="Game_ID", keep=False, inplace=True)
        self.games_info_10000.drop_duplicates(subset="steam_appid", keep=False, inplace=True)

        self.game_id_name['Game_Name'] = [str(i).lower() for i in self.game_id_name['Game_Name']]
        #self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].str.replace('-', ' ')
        #self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].str.replace(':', ' ')
        self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].str.replace('[{}]'.format(string.punctuation), '')
        self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].apply(lambda x: x.strip('®'))
        self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].apply(lambda x: x.strip('™'))

        self.games_info_10000['name'] = [str(i).lower() for i in self.games_info_10000['name']]
        #self.games_info_10000['name'] = self.games_info_10000['name'].str.replace('-', ' ')
        #self.games_info_10000['name'] = self.games_info_10000['name'].str.replace(':', ' ')
        self.games_info_10000['name'] = self.games_info_10000['name'].str.replace('[{}]'.format(string.punctuation), '')
        self.games_info_10000['name'] = self.games_info_10000['name'].apply(lambda x: x.strip('®'))
        self.games_info_10000['name'] = self.games_info_10000['name'].apply(lambda x: x.strip('™'))

        self.played_games_info = self.game_id_name.loc[self.game_id_name['Game_Name'].isin(self.games_info_10000['name'])]
        self.played_games_info = pd.merge(self.played_games_info, self.games_info_10000, left_on='Game_Name', right_on='name')

        self.game_id_name = self.game_id_name.loc[self.game_id_name['Game_Name'].isin(self.played_games_info['name'])]

        self.played_games_info.drop("steam_appid", inplace=True, axis=1)
        self.played_games_info.drop("name", inplace=True, axis=1)
        self.played_games_info.drop("dlc", inplace=True, axis=1)
        self.played_games_info.drop("header_image", inplace=True, axis=1)
        self.played_games_info.drop("pc_requirements", inplace=True, axis=1)
        self.played_games_info.drop("mac_requirements", inplace=True, axis=1)
        self.played_games_info.drop("linux_requirements", inplace=True, axis=1)
        self.played_games_info.drop("legal_notice", inplace=True, axis=1)
        self.played_games_info.drop("drm_notice", inplace=True, axis=1)
        self.played_games_info.drop("ext_user_account_notice", inplace=True, axis=1)
        self.played_games_info.drop("demos", inplace=True, axis=1)
        self.played_games_info.drop("price_overview", inplace=True, axis=1)
        self.played_games_info.drop("packages", inplace=True, axis=1)
        self.played_games_info.drop("package_groups", inplace=True, axis=1)
        self.played_games_info.drop("platforms", inplace=True, axis=1)
        self.played_games_info.drop("screenshots", inplace=True, axis=1)
        self.played_games_info.drop("movies", inplace=True, axis=1)
        self.played_games_info.drop("support_info", inplace=True, axis=1)
        self.played_games_info.drop("background", inplace=True, axis=1)
        self.played_games_info.drop("controller_support", inplace=True, axis=1)
        self.played_games_info.drop("fullgame", inplace=True, axis=1)
        self.played_games_info.drop("supported_languages", inplace=True, axis=1)
        self.played_games_info.drop("website", inplace=True, axis=1)
        self.played_games_info.drop("developers", inplace=True, axis=1)
        self.played_games_info.drop("publishers", inplace=True, axis=1)
        self.played_games_info.drop("metacritic", inplace=True, axis=1)
        self.played_games_info.drop("achievements", inplace=True, axis=1)
        self.played_games_info.drop("content_descriptors", inplace=True, axis=1)
        self.played_games_info.drop("release_date", inplace=True, axis=1)
        self.played_games_info.drop("reviews", inplace=True, axis=1)
        self.played_games_info.drop("required_age", inplace=True, axis=1)
        self.played_games_info.drop("is_free", inplace=True, axis=1)
        self.played_games_info.drop("recommendations", inplace=True, axis=1)
        self.played_games_info.drop("type", inplace=True, axis=1)

    def process_categories_and_genres(self):
        self.played_games_info = self.played_games_info[(self.played_games_info['categories'].notnull()) & (self.played_games_info['genres'].notnull())]

        for col in ['categories', 'genres']:
            self.played_games_info[col] = self.played_games_info[col].apply(lambda x: ','.join(item['description'] for item in literal_eval(x)))

    def plot_long_tail(self, dataframe, column_id, interaction_type):
        # long tail example
        fig = plt.figure(figsize=(15, 7))
        recmetrics.long_tail_plot(df=dataframe,
                                  item_id_column=column_id,
                                  interaction_type=interaction_type,
                                  percentage=0.5,
                                  x_labels=False)

    def plot_histogram(self, data, xlabel, title, average_value):
        print(data.max())
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.hist(data, bins=60)
        plt.axhline(y=average_value, color='r', linestyle='-')

    def analyze_datasets(self):

        print("\nDescription of users_games dataset:")
        print(self.users_games.describe())

        self.plot_long_tail(self.users_games, "Game_ID", "plays")

        # ploting frequency of games
        print("\n# of players that each game:\n{0}".format(self.users_games['Game_ID'].value_counts()))
        average_value = self.users_games['Game_ID'].value_counts().mean()
        self.plot_histogram(self.users_games['Game_ID'].value_counts(), '# of Players', 'Histogram of Games', average_value)

        # ploting frequency of users
        print("\n# of games that played by each user:\n{0}".format(self.users_games['User_ID'].value_counts()))
        average_value = self.users_games['User_ID'].value_counts().mean()
        self.plot_histogram(self.users_games['User_ID'].value_counts(), '# of Played Games', 'Histogram of Players', average_value)
        plt.show()

        # print("\nMin. rating: {0}".format(self.game_ratings['rating'].min()))
        # print("Max. rating: {0}".format(self.game_ratings['rating'].max()))

        # visualize the relationship between the total/average playtime of games and the number of players
        sns.jointplot(x='rating', y='number_of_player', data=self.game_ratings)
        sns.jointplot(x='total_hours', y='number_of_player', data=self.game_ratings)
        sns.jointplot(x='average_hours', y='number_of_player', data=self.game_ratings)
        plt.show()

        # create model (matrix of predicted values)
        # rating_matrix = self.game_ratings.pivot_table(index='average_hours', columns='number_of_player', values='rating').fillna(0)
        # sns.heatmap(rating_matrix, annot=True)















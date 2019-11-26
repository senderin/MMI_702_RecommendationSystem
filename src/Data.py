import re
import string
import seaborn as sns
import pandas as pd
import recmetrics
import matplotlib.pyplot as plt
from ast import literal_eval

from pandas import array


class Data():

    __instance = None
    playtime = pd.DataFrame()
    game_id_name = pd.DataFrame()
    users_games = pd.DataFrame()
    game_ratings = pd.DataFrame()
    steam_app_data = pd.DataFrame()
    played_games = pd.DataFrame()

    @staticmethod
    def get_instance():
        if Data.__instance == None:
            Data()
        return Data.__instance

    def __init__(self):
        if Data.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Data.__instance = self

        self.read_csvs()
        self.preprocess_steam_app_data()
        self.preprocess_game_id_name()
        self.preprocess_playtime()

        self.played_games = self.get_played_games_df()
        self.users_games = self.get_users_games_df()
        self.game_ratings = self.calculate_game_ratings()

    def read_csvs(self):
        # reading datasets
        self.game_id_name = pd.read_csv('game_id_name.csv')
        self.steam_app_data = pd.read_csv('steam_app_data.csv')
        self.playtime = pd.read_csv('playtime.csv')

    def preprocess_steam_app_data(self):
        self.steam_app_data.rename(columns={'name': 'Game_Name'}, inplace=True)
        self.steam_app_data.rename(columns={'steam_appid': 'Game_ID'}, inplace=True)
        self.steam_app_data.drop_duplicates(subset="Game_Name", keep=False, inplace=True)
        self.steam_app_data.drop_duplicates(subset="Game_ID", keep=False, inplace=True)
        self.steam_app_data = self.steam_app_data.drop(['Game_ID'], axis=1)
        self.steam_app_data = self.process_categories_and_genres(self.steam_app_data)
        self.preprocessing_column(self.steam_app_data, 'Game_Name')

    def preprocess_game_id_name(self):
        self.game_id_name.drop_duplicates(subset="Game_Name", keep=False, inplace=True)
        self.game_id_name.drop_duplicates(subset="Game_ID", keep=False, inplace=True)
        self.preprocessing_column(self.game_id_name, 'Game_Name')
        self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].apply(lambda x: x.strip('®'))
        self.game_id_name['Game_Name'] = self.game_id_name['Game_Name'].apply(lambda x: x.strip('™'))
        # remove games that not consisted in steam_app_data
        self.game_id_name = self.game_id_name.loc[self.game_id_name['Game_Name'].isin(self.steam_app_data['Game_Name'])]
        # remove games that not played
        self.game_id_name = self.game_id_name.loc[self.game_id_name['Game_ID'].isin(self.playtime['Game_ID'])]

    def preprocess_playtime(self):
        # eliminate ones that have no corresponding name
        self.playtime = self.playtime.loc[self.playtime['Game_ID'].isin(self.game_id_name['Game_ID'])]

    def get_users_games_df(self):
        temp = pd.merge(self.game_id_name, self.playtime, on='Game_ID')
        temp = temp.groupby('User_ID').filter(lambda x: len(x) > 10)
        return temp

    def get_played_games_df(self):
        played_games = self.game_id_name.loc[self.game_id_name['Game_Name'].isin(self.steam_app_data['Game_Name'])]
        temp = self.steam_app_data.copy()
        played_games = pd.merge(played_games, temp, on='Game_Name')
        return played_games

    def preprocessing_column(self, df, column_name):
        df[column_name] = df[column_name].fillna('')
        # Convert text to lowercase
        df[column_name] = [str(i).lower() for i in df[column_name]]
        # Remove numbers
        # df[column_name] = [re.sub(r'\d+', '', str(i)) for i in df[column_name]]
        # Remove whitespaces
        df[column_name] = [str(i).strip() for i in df[column_name]]
        # Remove html tags
        df[column_name] = [self.clean_html_tags(str(i)) for i in df[column_name]]
        # Remove punctuation
        df[column_name] = [str(i).translate(str.maketrans('', '', string.punctuation)) for i in df[column_name]]
        df.reset_index()
        return df

    def clean_html_tags(self, text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', text)
        return cleantext

    def calculate_game_ratings(self):
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

    def process_categories_and_genres(self, df):
        df = df.copy()
        df = df[(df['categories'].notnull()) & (df['genres'].notnull())]

        for col in ['categories', 'genres']:
            df[col] = df[col].apply(lambda x: ';'.join(item['description'] for item in literal_eval(x)))

        return df

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
        user_ids = self.users_games['User_ID'].unique()
        print("users_games # user {0}".format(len(user_ids)))
        game_ids = self.users_games['Game_ID'].unique()
        print("users_games # game {0}".format(len(game_ids)))

        print("played_games # game {0}".format(len(self.played_games['Game_ID'].unique())))
        print("game_id_name # game {0}".format(len(self.game_id_name['Game_ID'].unique())))
        print("playtime # game {0}".format(len(self.playtime['Game_ID'].unique())))
        print("game_ratings # game {0}".format(len(self.game_ratings['Game_ID'].unique())))

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



#data = Data.get_instance()
#print(data.playtime)
#print()
#print(data.game_id_name[['Game_ID', 'Game_Name']])
#print()
#print(data.played_games[['Game_ID', 'Game_Name']])
#print()
#print(data.users_games[['Game_ID', 'Game_Name']])
#print()
#print(data.game_ratings[['Game_ID', 'Game_Name']])
#print()
#print(data.steam_app_data[['Game_ID', 'Game_Name']])
#data.analyze_datasets()












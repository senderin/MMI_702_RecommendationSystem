import random
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.Data import Data
from src.Evalution import Evaluation
import matplotlib.pyplot as plt


class ContentBasedRecommendation():
    data = None
    playtime = None
    game_id_name = None
    steam_app_data = None
    played_games = None

    def __init__(self):
        self.data = Data.get_instance()
        self.playtime = self.data.playtime.copy()
        self.game_id_name = self.data.game_id_name.copy()
        self.steam_app_data = self.data.steam_app_data.copy()
        self.played_games = self.data.played_games.copy()

    def preprocess(self, df, column_name):
        #print('shape before process {0}'.format(df.shape))
        df.reset_index()
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
        #print('shape after process {0}'.format(df.shape))
        # print(df.head(10))

    def clean_html_tags(self, text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', text)
        return cleantext

    def recommend_for_game(self, game_name, metadata_name):
        df = self.game_id_name.loc[self.game_id_name['Game_Name'].isin(self.steam_app_data['Game_Name'])]
        # print('shape after intersection {0}'.format(df.shape))
        df = pd.merge(df, self.steam_app_data, on='Game_Name')
        # print('shape after merge {0}'.format(df.shape))

        game_name = game_name.lower().replace('[{}]'.format(string.punctuation), ' ').strip('®').strip('™')
        index = df.loc[df['Game_Name'] == game_name].index[0]
        game_id = df.loc[df['Game_Name'] == game_name]['Game_ID']
        column_name = metadata_name
        self.preprocess(df, column_name)

        count_vect = CountVectorizer(stop_words="english", ngram_range=(1,3))
        train_counts = count_vect.fit_transform(df[column_name])

        tfidf_transformer = TfidfTransformer()
        train_tfidf = tfidf_transformer.fit_transform(train_counts)

        neigh = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=11, n_jobs=-1)
        neigh.fit(train_tfidf)

        test = df.loc[[index]]
        test_counts = count_vect.transform(test[column_name])
        test_tfidf = tfidf_transformer.transform(test_counts)
        distances, indices = neigh.kneighbors(test_tfidf)

        names_similar = pd.Series(indices.flatten()).map(df['Game_Name']).values.tolist()
        result = pd.DataFrame({'distance': distances.flatten(), 'index': indices.flatten(), 'name': names_similar})
        names_similar.remove(game_name)
        ids_similar = self.data.played_games.loc[self.data.played_games['Game_Name'].isin(names_similar)]['Game_ID'].values.tolist()
        result = result.iloc[1:]
        print(metadata_name)
        print(result)

        return ids_similar, names_similar

    def recommend_with_all_metadata(self, played_games):
        all_text_data = self.text_data_of_games(self.data.played_games)
        self.preprocess(all_text_data, 'text')
        played_text_data = self.text_data_of_games(played_games)
        self.preprocess(played_text_data, 'text')

        indexes = []
        for index, row in played_games.iterrows():
            index = self.data.played_games.loc[self.data.played_games['Game_Name'] == row['Game_Name']].index[0]
            indexes.append(index)

        count_vect = CountVectorizer(stop_words="english", ngram_range=(1,3))
        train_counts = count_vect.fit_transform(all_text_data['text'])

        tfidf_transformer = TfidfTransformer()
        train_tfidf = tfidf_transformer.fit_transform(train_counts)

        neigh = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=11, n_jobs=-1)
        neigh.fit(train_tfidf)

        results = pd.DataFrame()
        for i in range(len(indexes)):
            test = played_text_data.loc[[i]]
            test_counts = count_vect.transform(test['text'])
            test_tfidf = tfidf_transformer.transform(test_counts)
            features = count_vect.get_feature_names()

            distances, indices = neigh.kneighbors(test_tfidf)

            names = pd.Series(indices.flatten()).map(self.data.played_games['Game_Name']).values.tolist()
            ids = self.data.played_games.loc[self.data.played_games['Game_Name'].isin(names)]['Game_ID']
            result = pd.DataFrame({'distance': distances.flatten(), 'index': indices.flatten(), 'name': names})
            result = result.iloc[1:]
            results = results.append(result, ignore_index=True)
            print('all_metadata')
            print(results.head(10))

        if not len(results) == 0:
            predictions_name = results.sample(n=10)['name'].values.tolist()
            predictions_id = self.data.played_games.loc[self.data.played_games['Game_Name'].isin(predictions_name)]['Game_ID'].values.tolist()
        else:
            predictions_id = [0] * 10
            predictions_name = [''] * 10

        return predictions_id, predictions_name

    def text_data_of_games(self, games_info):
        text = []
        for index, game in games_info.iterrows():
            line = str(game["short_description"] + game["about_the_game"] + game["detailed_description"])
            line = line + str(game['genres'] + game['categories']).replace(';', ' ')
            text.append(line)
            #print(line)

        temp = pd.DataFrame()
        temp['text'] = text
        #print(temp.head())
        return temp

    def plot_tfidf(self, tfidf_matrix, features, row):
        # Create a series from the sparse matrix
        d = pd.Series(tfidf_matrix.getrow(row).toarray().flatten(), index=features).sort_values(ascending=False)
        print(d.head(10))

        ax = d[:20].plot(kind='bar', title='TF-IDF Values',
                         figsize=(10, 6), width=.8, fontsize=14, rot=90)
        ax.title.set_size(20)
        plt.show()

    def plot_word_frequencies(self, matrix, features):
        # Create a series from the sparse matrix
        d = pd.Series(matrix.toarray().flatten(),
                      index=features).sort_values(ascending=False)

        ax = d[:10].plot(kind='bar', figsize=(10, 6), width=.8, fontsize=14, rot=45,
                         title='Word Counts')
        ax.title.set_size(18)
        plt.show()

#   def word_count_matrix(self, df):
#       # Extract text for a particular person
#       text = df.loc[df['Game_Name'] == self.game_name]
#       # Define the count vectorizer that will be used to process the data
#       count_vectorizer = CountVectorizer(stop_words = "english")
#       # Apply this vectorizer to text to get a sparse matrix of counts
#       count_matrix = count_vectorizer.fit_transform(text[self.column_name])
#       # Get the names of the features
#       features = count_vectorizer.get_feature_names()
#       # Create a series from the sparse matrix
#       d = pd.Series(count_matrix.toarray().flatten(),
#                     index=features).sort_values(ascending=False)

#       ax = d[:10].plot(kind='bar', figsize=(10, 6), width=.8, fontsize=14, rot=90,
#                        title='Word Counts')
#       ax.title.set_size(18)
#       plt.show()

#       return count_matrix














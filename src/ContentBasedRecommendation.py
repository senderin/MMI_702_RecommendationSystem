import re
import string
from collections import Counter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.Data import Data
import matplotlib.pyplot as plt


class ContentBasedRecommendation():
    data = None
    playtime = None
    game_id_name = None
    steam_app_data = None

    def __init__(self):
        self.data = Data.get_instance()
        self.playtime = self.data.playtime.copy()
        self.game_id_name = self.data.game_id_name.copy()
        self.steam_app_data = self.data.steam_app_data.copy()

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

        print()
        #game_name = 'portal 2'
        game_name = game_name.lower().replace('[{}]'.format(string.punctuation), ' ').strip('®').strip('™')
        index = df.loc[df['Game_Name'] == game_name].index[0]
        column_name = metadata_name
        self.preprocess(df, column_name)

        count_vect = CountVectorizer(stop_words="english")
        X_train_counts = count_vect.fit_transform(df[column_name])

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        neigh = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=10, n_jobs=-1)
        neigh.fit(X_train_tfidf)

        test = df.loc[[index]]
        X_test_counts = count_vect.transform(test[column_name])
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        distances, indices = neigh.kneighbors(X_test_tfidf)

        names_similar = pd.Series(indices.flatten()).map(df['Game_Name']).values.tolist()
        ids_similar = self.data.played_games.loc[self.data.played_games['Game_Name'].isin(names_similar)]['Game_ID']
        result = pd.DataFrame({'distance': distances.flatten(), 'index': indices.flatten(), 'name': names_similar})
        result = result.iloc[1:]
        # print(result)

        return ids_similar, names_similar

    def recommend_for_user_set(self, test, type_name, is_plot):
        pass

    def recommend_for_user(self, played_games, column_name):
        pool = self.create_pool(most_played_games, cosine_sim)
        if not len(pool) == 0:
            predictions_name = random.sample(pool, 10)
            predictions_id = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(predictions_name)]['Game_ID'].values.tolist()
        else:
            predictions_id = [0] * 10

        return predictions_id
        return self.recommend_for_game('portal 2', column_name)

    def recommend_with_all_metadata(self, played_games):
        all_text_data = self.text_data_of_games(self.data.played_games)
        self.preprocess(all_text_data, 'text')
        played_text_data = self.text_data_of_games(played_games)
        self.preprocess(played_text_data, 'text')

        indexes = []
        for index, row in played_games.iterrows():
            index = self.data.played_games.loc[self.data.played_games['Game_Name'] == row['Game_Name']].index[0]
            indexes.append(index)

        count_vect = CountVectorizer(stop_words="english")
        X_train_counts = count_vect.fit_transform(all_text_data['text'])

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        neigh = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=11, n_jobs=-1)
        neigh.fit(X_train_tfidf)

        results = pd.DataFrame()
        for i in range(len(indexes)):
            test = played_text_data.loc[[i]]
            X_test_counts = count_vect.transform(test['text'])
            X_test_tfidf = tfidf_transformer.transform(X_test_counts)
            distances, indices = neigh.kneighbors(X_test_tfidf)

            names = pd.Series(indices.flatten()).map(self.data.played_games['Game_Name']).values.tolist()
            ids = self.data.played_games.loc[self.data.played_games['Game_Name'].isin(names)]['Game_ID']
            result = pd.DataFrame({'distance': distances.flatten(), 'index': indices.flatten(), 'name': names})
            result = result.iloc[1:]
            results = results.append(result, ignore_index=True)

        if not len(results) == 0:
            predictions_name = results.sample(n=10)['name'].values.tolist()
            predictions_id = self.data.played_games.loc[self.data.played_games['Game_Name'].isin(predictions_name)]['Game_ID'].values.tolist()
        else:
            predictions_id = [0] * 10
            predictions_name = [''] * 10

        print(predictions_name)

        return predictions_id, predictions_name

    def text_data_of_games(self, games_info):
        text = []
        for index, game in games_info.iterrows():
            line = game["short_description"] + game["about_the_game"] + game["detailed_description"]
            text.append(line)
            #print(line)

        temp = pd.DataFrame()
        temp['text'] = text
        #print(temp.head())
        return temp

#   def recommend_for_game(self, game_name, column_name):
#       self.game_name = game_name
#       self.column_name = column_name

#       print()
#       print('the game name: {0}'.format(self.game_name))
#       index = self.played_games_info.loc[self.played_games_info['Game_Name'] == self.game_name].index[0]
#       print('index of the game in original dataframe: {0}'.format(index))
#       #print(self.played_games_info[index - 5:index+5]['Game_Name'])
#       #print(self.played_games_info.tail(10)['Game_Name'])

#       self.dfs.games_info_10000.rename(columns={'name': 'Game_Name'}, inplace=True)
#       self.dfs.games_info_10000.rename(columns={'steam_appid': 'Game_ID'}, inplace=True)
#       df = self.preprocessing_column(self.played_games_info)

#       # word_count_matrix = self.word_count_matrix(df)
#       tfidf_matrix, features, pairwise_similarity = self.tfidf_matrix(df)
#       print('the shape of tfidf matrix: {0}'.format(tfidf_matrix.shape))

#       # show tfidf matrix
#       corpus_index = [n for n in df['Game_Name']]
#       temp_df = pd.DataFrame(tfidf_matrix.todense(), index=corpus_index, columns=features)
#       # print(temp_df[index - 5:index+5])
#       #print(temp_df.tail(10))

#       row = df.loc[df['Game_Name'] == self.game_name].index[0]
#       print(tfidf_matrix.getrow(row))
#       indices = np.nonzero(pairwise_similarity[65] > 0)
#       print(indices)
#       for i in range(len(indices)):
#            print(pairwise_similarity[65][indices[i]])
#            print(df['Game_Name'][indices[i]])

#       feature_index = tfidf_matrix[row, :].nonzero()[1]
#       print(feature_index)
#       tfidf_scores = zip(feature_index, [tfidf_matrix[row, x] for x in feature_index])
#       for w, s in [(features[i], s) for (i, s) in tfidf_scores]:
#           print(w, s)

#       knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(tfidf_matrix)
#       result = self.get_closest_neighs(knn, tfidf_matrix, df)
#       print(result)

#   def preprocessing_column(self, df):
#       print('Before preprocessing:')
#       #print(df[self.column_name].tail(20))
#       print(df.shape)
#       df[self.column_name] = df[self.column_name].fillna('')
#       # Convert text to lowercase
#       df[self.column_name] = [str(i).lower() for i in df[self.column_name]]
#       # Remove numbers
#       # df[self.column_name] = [re.sub(r'\d+', '', str(i)) for i in df[self.column_name]]
#       # Remove whitespaces
#       df[self.column_name] = [str(i).strip() for i in df[self.column_name]]
#       # Remove html tags
#       df[self.column_name] = [self.clean_html_tags(str(i)) for i in df[self.column_name]]
#       # Remove punctuation
#       df[self.column_name] = [str(i).translate(str.maketrans('', '', string.punctuation)) for i in df[self.column_name]]
#       print('\nAfter preprocessing:')
#       #print(df[self.column_name].tail(20))
#       print(df.shape)
#       return df

#   def clean_html_tags(self, text):
#       cleanr = re.compile('<.*?>')
#       cleantext = re.sub(cleanr, ' ', text)
#       return cleantext


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

#   def plot_tfidf(self, tfidf_matrix, features, row):
#       # Create a series from the sparse matrix
#       d = pd.Series(tfidf_matrix.getrow(row).toarray().flatten(), index=features).sort_values(ascending=False)
#       print(d.head(10))

#       ax = d[:20].plot(kind='bar', title='TF-IDF Values',
#                        figsize=(10, 6), width=.8, fontsize=14, rot=90)
#       ax.title.set_size(20)
#       plt.show()

#   def tfidf_matrix(self, df):
#       # Define the TFIDF vectorizer that will be used to process the data
#       tfidf_vectorizer = TfidfVectorizer(stop_words = "english")
#       # Apply this vectorizer to the full dataset to create normalized vectors
#       tfidf_matrix = tfidf_vectorizer.fit_transform(df[self.column_name])
#       # Get the names of the features
#       features = tfidf_vectorizer.get_feature_names()
#       #print(features)
#       pairwise_similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
#       np.fill_diagonal(pairwise_similarity, 0)
#       print(pairwise_similarity)

#       return tfidf_matrix, features, pairwise_similarity

#   def get_closest_neighs(self, knn, tfidf_matrix, df):
#       row = df.loc[df['Game_Name'] == self.game_name].index[0]

#       distances, indices = knn.kneighbors(tfidf_matrix.getrow(row))
#       print(indices.flatten())
#       for i in range(0, len(distances.flatten())):
#           if (i == 0):
#               print('recommended for : {0} -- {1}'.format(df.index[row],
#                                                           df['Game_Name'][indices.flatten()[i]]))
#           else:
#               print("{0} -- {1} -- {2}".format(df.index[indices.flatten()[i]],
#                                                df['Game_Name'][indices.flatten()[i]], distances.flatten()[i]))

#       distances, indices = knn.kneighbors(tfidf_matrix.getrow(row))
#       print(indices.flatten())
#       names_similar = pd.Series(indices.flatten()).map(df['Game_Name'])
#       print(df['Game_Name'][indices.flatten()])
#       ids_similar = pd.Series(indices.flatten()).map(df['Game_ID'])
#       result = pd.DataFrame({'distance': distances.flatten(), 'index': indices.flatten(), 'name': names_similar})

#       return result

    # most played game name kısmı
    # def KNN_4(self, tfidf, user_id):
    #
    #    user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == 24]).reset_index()
    #    game_ids = user_info['Game_ID'].tolist()
    #    game_names = user_info['Game_Name'].tolist()
    #    game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
    #    cat_list = game_info['categories'].tolist()
    #    genre_list = game_info['genres'].tolist()
    #
    #    game_name = user_info.sort_values(by=['Hours']).tail(1)
    #    print(game_name)

    ## takes in game title as input and returns the top 10 recommended games
    #def recommend_game_tfidf(self, name, cosine_sim):
    #    # initializing the empty list of recommended games
    #    recommendations = []
#
    #    infos = self.played_games_info.loc[self.played_games_info['Game_Name'] == name]
#
    #    if not infos.empty == True:
    #        # getting the index of the game that matches the title
    #        idx = self.played_games_info.loc[self.played_games_info['Game_Name'] == name].index[0]
#
    #        # creating a Series with the similarity scores in descending order
    #        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
#
    #        # getting the indexes of the 10 most similar games
    #        top_10_indexes = list(score_series.iloc[1:11].index)
#
    #        # populating the list with the titles of the best 10 matching games
    #        for i in top_10_indexes:
    #            recommendations.append(list(self.played_games_info['Game_Name'])[i])
#
    #        for i in range(len(recommendations)):
    #            names = self.played_games_info.loc[self.played_games_info['Game_ID'] == recommendations[i]]['Game_Name']
    #            # print(name)
#
    #    return recommendations
#

    #def create_count_matrix(self, column_name):
    #    # new columns
    #    self.played_games_info['keywords'] = ""
    #    self.played_games_info['bag_of_words'] = ""
#
    #    for index, row in self.played_games_info.iterrows():
    #        row = row.copy()
    #        column_info = row[column_name]
#
    #        # uses english stopwords from NLTK and discards all punctuation characters as well
    #        r = Rake()
#
    #        # extracting the words by passing the text
    #        r.extract_keywords_from_text(str(column_info))
#
    #        # getting the dictionary with key words as keys and their scores as values
    #        key_words_dict_scores = r.get_word_degrees()
#
    #        # assigning the key words to the new column for the corresponding movie
    #        row['keywords'] = list(key_words_dict_scores.keys())
    #        self.played_games_info.at[index, 'keywords'] = row['keywords']
    #        self.played_games_info.at[index, 'bag_of_words'] = " ".join(row['keywords'])
#
    #    # instantiating and generating the count matrix
    #    count_vectorizer = CountVectorizer(stop_words = 'english')
    #    count_matrix = count_vectorizer.fit_transform(self.played_games_info['bag_of_words'])
    #    # Get the names of the features
    #    features = count_vectorizer.get_feature_names()
    #    # generating the cosine similarity matrix
    #    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    #    self.plot_word_frequencies(count_matrix, features)
#
    #    return cosine_sim, count_matrix, features
#
    #def create_tfidf_matrix(self, column_name):
    #    self.played_games_info[column_name] = self.played_games_info[column_name].fillna('')
#
    #    tfidf_vectorizer = TfidfVectorizer(min_df = 0, max_df=0.5, stop_words = "english", ngram_range = (1,3))
    #    tfidf_matrix = tfidf_vectorizer.fit_transform(self.played_games_info[column_name])
#
    #    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#
    #    features = tfidf_vectorizer.get_feature_names()
    #    indices = zip(*tfidf_matrix.nonzero())
    #    #for row, column in indices:
    #    #    print('({0}, {1}) {2}' .format(row, features[column], tfidf_matrix[row, column]))
#
    #    # self.plot_word_frequencies(tfidf_matrix, features)
    #    return cosine_sim, tfidf_matrix, features
#
    #def plot_word_frequencies(self, matrix, features):
    #    # Create a series from the sparse matrix
    #    d = pd.Series(matrix.toarray().flatten(),
    #                  index=features).sort_values(ascending=False)
#
    #    ax = d[:10].plot(kind='bar', figsize=(10, 6), width=.8, fontsize=14, rot=45,
    #                     title='Word Counts')
    #    ax.title.set_size(18)
#
    #def get_recommendations_for_a_user(self, user_id, is_plot, cosine_sim):
    #    user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == user_id])
    #    game_ids = user_info['Game_ID'].tolist()
    #    game_names = user_info['Game_Name'].tolist()
    #    game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
    #    cat_list = game_info['categories'].tolist()
    #    genre_list = game_info['genres'].tolist()
#
    #    if is_plot == True:
    #        self.plot_user_preferences()
#
    #    most_played_games = user_info.sort_values(by=['Hours']).tail(1)
#
    #    pool = self.create_pool(most_played_games, cosine_sim)
    #    if not len(pool) == 0:
    #        predictions_name = random.sample(pool, 10)
    #        predictions_id = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(predictions_name)]['Game_ID'].values.tolist()
    #    else:
    #        predictions_id = [0] * 10
#
    #    return predictions_id
#

    #def recommend_for_user_set(self, test, column_name, is_plot):
    #    cosine_sim, matrix, features = self.create_tfidf_matrix(column_name)
    #    total = len(test)
    #    recommendations = []
    #    for index, user in test.iterrows():
    #        # print('{0}/{1}'.format(index, total))
    #        user_id = user['User_ID']
    #        predictions = self.get_recommendations_for_a_user(user_id, is_plot, cosine_sim)
    #        recommendations.append(predictions)
#
    #    return recommendations
#
    #def user_analysis(self, user_id):
    #    user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == user_id])
    #    game_ids = user_info['Game_ID'].tolist()
    #    game_names = user_info['Game_Name'].tolist()
    #    game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
    #    cat_list = game_info['categories'].tolist()
    #    genre_list = game_info['genres'].tolist()
#
#
    #    user_info = pd.merge(user_info, self.dfs.game_ratings, on ='Game_ID')
    #    user_info['user_rating'] = user_info['Hours']/user_info['average_hours']
    #    user_info['user_rating_2'] = user_info['Hours']/user_info['Hours'].mean()
    #    # user_rating ve user_rating_2 paralel
    #    user_info['user_rating_3'] = user_info['user_rating']*user_info['user_rating_2']
    #    user_info.sort_values(by=['user_rating'])
#
    #    #sns.jointplot(x='user_rating', y='user_rating_3', data=user_info)
    #    #plt.show()
#
    #    # for genre
    #    self.plot_word_cloud(str(genre_list).replace("'", '').replace('[', '').replace(']', '').split(','))
#
    #    # for categories
    #    self.plot_word_cloud(str(cat_list).replace("'", '').replace('[', '').replace(']', '').split(','))
#
    #    print(user_info.head(10))
#

#
    #def recommend_for_game(self, name):
    #    # initializing the empty list of recommended games
    #    recommendations = []
#
    #    info = self.played_games_info.loc[self.played_games_info['Game_Name'] == name]
#
    #    if not info.empty == True:
    #        # getting the index of the game that matches the title
    #        idx = self.played_games_info.loc[self.played_games_info['Game_Name'] == name].index[0]
#
    #        score_serie = self.KNN(name)
#
    #        # getting the indexes of the 10 most similar games
    #        top_10_indexes = list(score_serie.iloc[1:11].index)
#
    #        # populating the list with the titles of the best 10 matching games
    #        for i in top_10_indexes:
    #            recommendations.append(list(self.played_games_info['Game_Name'])[i])
#
    #        for i in range(len(recommendations)):
    #            names = self.played_games_info.loc[self.played_games_info['Game_ID'] == recommendations[i]]['Game_Name']
    #            # print(name)
#
    #    return recommendations
#
    #def KNN(self, name):
    #    return []



    #def KNN(self):
    #    games = self.dfs.game_ratings.copy()
    #    games.drop('Game_Name', inplace = True, axis = 1)
#
    #    min_max_scaler = MinMaxScaler()
    #    games_features = min_max_scaler.fit_transform(games)
    #    np.round(games_features, 2)
    #    nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(games_features)
    #    distances, indices = nbrs.kneighbors(games_features)
    #    return distances, indices
#

#
    #def get_index_from_name(self, name):
    #    index = self.dfs.game_ratings.index[self.dfs.game_ratings['Game_Name'] == str(name).lower()].tolist()
    #    return index
#
    #def get_recommendation_with_knn(self, name):
    #    self.dfs.game_ratings = self.dfs.game_ratings.reset_index(drop = True)
#
    #    distances, indices = self.KNN()
#
    #    # query_index = self.get_index_from_name('mass effect')
    #    query_index = np.random.choice(self.dfs.game_ratings.shape[0])
    #    print('for {0}'.format(self.dfs.game_ratings['Game_Name'][query_index]))
    #    for query_index in indices[query_index][1:]:
    #        print(self.dfs.game_ratings['Game_Name'][query_index])
#
    #def get_recommend_for_user_with_knn(self, user_id):
    #    user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == user_id])
    #    game_names = user_info['Game_Name'].tolist()
    #    game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
    #    return game_info
#
#
#
    #def KNN_2(self, tfidf_matrix, predict_matrix):
    #    NUMBER_OF_CLUSTERS = 10
    #    km = KMeans(
    #        n_clusters=NUMBER_OF_CLUSTERS,
    #        init='k-means++',
    #        max_iter=500)
    #    km.fit(tfidf_matrix)
#
    #    clusters = km.predict(predict_matrix)
#
    #    return km, clusters
#
    #def KNN_predict_2(self, km, matrix):
#
    #    # First: for each we get its corresponding cluster
    #    clusters = km.predict(matrix)
#
    #    return clusters
#
    #def PCA(self, tfidf_matrix):
    #    # We train the PCA on the dense version of the tf-idf.
    #    pca = PCA(n_components=3)
    #    two_dim = pca.fit_transform(tfidf_matrix.todense())
#
    #    scatter_x = two_dim[:, 0]  # first principle component
    #    scatter_y = two_dim[:, 1]  # second principle component
    #    scatter_z = two_dim[:, 2]  # second principle component
#
    #    return scatter_x, scatter_y, scatter_z
#
    #def plot_tfidf_matrix(self, clusters, km, scatter_x, scatter_y, scatter_z):
    #    plt.style.use('ggplot')
#
    #    fig = plt.figure(figsize=(5.5, 3))
    #    ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
#
    #    #fig, ax = plt.subplots()
    #    fig.set_size_inches(20, 10)
#
    #    # color map for NUMBER_OF_CLUSTERS we have
    #    labels_color_map = {
    #        0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
    #        5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
    #    }
#
    #    # group by clusters and scatter plot every cluster
    #    # with a colour and a label
    #    for group in np.unique(clusters):
    #        ix = np.where(clusters == group)
    #        ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=labels_color_map[group], label=group)
#
    #    centroids = km.cluster_centers_
    #    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=160, c='red')
#
    #    ax.legend()
    #    plt.xlabel("PCA 0")
    #    plt.ylabel("PCA 1")
    #    plt.zlabel("PCA 2")
    #    plt.show()
#
    #def print_top10words_in_clusters(self, km, features):
    #    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
    #    for i in range(10):
    #        print("Cluster %d:" % i, end='')
    #        for ind in order_centroids[i, :10]:
    #            print(' %s' % features[ind], end='')
    #        print()
#
    #def KNN_3(self, list1, list2, column_name):
    #    vectorizer = TfidfVectorizer(min_df=0, max_df=0.5, stop_words="english", ngram_range=(1, 3))
    #    vec = vectorizer.fit(list1[column_name])  # train vec using list1
    #    vectorized = vec.transform(list1[column_name])  # transform list1 using vec
#
    #    km = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, precompute_distances=True,
    #                verbose=0, random_state=None, n_jobs=1)
    #    features = vectorizer.get_feature_names()
    #    km.fit(vectorized)
    #    list2Vec = vec.transform(list2[column_name])  # transform list2 using vec
    #    clusters = km.predict(list2Vec)
#
    #    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
    #    for i in range(10):
    #        print("Cluster %d:" % i, end='')
    #        for ind in order_centroids[i, :10]:
    #            print(' %s' % features[ind], end='')
    #        print()












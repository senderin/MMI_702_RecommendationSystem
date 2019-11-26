import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.Data import Data

from src.Evalution import Evaluation


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

        count_vect = CountVectorizer(stop_words="english")
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
        # print(result)

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

        count_vect = CountVectorizer(stop_words="english")
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
            distances, indices = neigh.kneighbors(test_tfidf)

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

#
    #def plot_word_frequencies(self, matrix, features):
    #    # Create a series from the sparse matrix
    #    d = pd.Series(matrix.toarray().flatten(),
    #                  index=features).sort_values(ascending=False)
#
    #    ax = d[:10].plot(kind='bar', figsize=(10, 6), width=.8, fontsize=14, rot=45,
    #                     title='Word Counts')
    #    ax.title.set_size(18)

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













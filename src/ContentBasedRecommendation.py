from collections import Counter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
from src.Dataframes import Dataframes
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


class ContentBasedRecommendation():

    games_info = None
    users_games = None
    dfs = None
    played_games_info = None

    def __init__(self):
        self.dfs = Dataframes.get_instance()
        self.games_info = self.dfs.played_games_info
        self.users_games = self.dfs.users_games
        self.played_games_info = self.dfs.played_games_info


    # takes in game title as input and returns the top 10 recommended games
    def recommend_for_a_game(self, name, cosine_sim):
        # initializing the empty list of recommended games
        recommendations = []

        infos = self.games_info.loc[self.games_info['Game_Name'] == name]

        if not infos.empty == True:
            # getting the index of the game that matches the title
            idx = self.games_info.loc[self.games_info['Game_Name'] == name].index[0]

            # creating a Series with the similarity scores in descending order
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

            # getting the indexes of the 10 most similar games
            top_10_indexes = list(score_series.iloc[1:11].index)

            # populating the list with the titles of the best 10 matching games
            for i in top_10_indexes:
                recommendations.append(list(self.games_info['Game_Name'])[i])

            for i in range(len(recommendations)):
                names = self.games_info.loc[self.games_info['Game_ID'] == recommendations[i]]['Game_Name']
                # print(name)

        return recommendations

    def create_pool(self, most_played_games, cosine_sim):
        recommendationPool = []
        for index, row in most_played_games.iterrows():
            contentbased_predictions = self.recommend_for_a_game(row['Game_Name'], cosine_sim)
            if not len(contentbased_predictions) == 0:
                recommendationPool.extend(contentbased_predictions)
        # print(recommendationPool)
        return recommendationPool

    def create_count_matrix(self, column_name):
        # new columns
        self.games_info['keywords'] = ""
        self.games_info['bag_of_words'] = ""

        for index, row in self.games_info.iterrows():
            row = row.copy()
            short_description = row[column_name]

            # uses english stopwords from NLTK and discards all punctuation characters as well
            r = Rake()

            # extracting the words by passing the text
            r.extract_keywords_from_text(str(short_description))

            # getting the dictionary with key words as keys and their scores as values
            key_words_dict_scores = r.get_word_degrees()

            # assigning the key words to the new column for the corresponding movie
            row['keywords'] = list(key_words_dict_scores.keys())
            self.games_info.at[index, 'keywords'] = row['keywords']
            self.games_info.at[index, 'bag_of_words'] = " ".join(row['keywords'])

        # instantiating and generating the count matrix
        count = CountVectorizer(stop_words = 'english')
        count_matrix = count.fit_transform(self.games_info['bag_of_words'])

        ## generating the cosine similarity matrix
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        return count_matrix, cosine_sim

    def create_tfidf_matrix(self, column_name, df):
        df[column_name] = df[column_name].fillna('')

        tfidf_vectorizer = TfidfVectorizer(min_df = 0, max_df=0.5, stop_words = "english", charset_error = "ignore", ngram_range = (1,3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_name])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        features = tfidf_vectorizer.get_feature_names()
        indices = zip(*tfidf_matrix.nonzero())
        for row, column in indices:
            print('({0}, {1}) {2}' .format(row, features[column], tfidf_matrix[row, column]))

        return cosine_sim, tfidf_matrix, features


    def get_recommendations_with_tfidf(self, column_name, name):
        cosine_sim = self.create_tfidf_matrix(column_name)
        ids = self.recommend_for_a_game(name, cosine_sim)
        print(ids)

    def get_recommendations_with_count_matrix(self, column_name, name):
        count_matrix, cosine_sim = self.create_count_matrix(column_name)
        ids = self.recommend_for_a_game(name, cosine_sim)
        print(ids)

    def get_recommendations_for_a_user(self, user_id, is_plot, cosine_sim):
        user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == user_id])
        game_ids = user_info['Game_ID'].tolist()
        game_names = user_info['Game_Name'].tolist()
        game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
        cat_list = game_info['categories'].tolist()
        genre_list = game_info['genres'].tolist()

        if is_plot == True:
            self.plot_user_preferences()

        most_played_games = user_info.sort_values(by=['Hours']).tail(1)

        pool = self.create_pool(most_played_games, cosine_sim)
        if not len(pool) == 0:
            predictions_name = random.sample(pool, 10)
            predictions_id = self.games_info.loc[self.games_info['Game_Name'].isin(predictions_name)]['Game_ID'].values.tolist()
        else:
            predictions_id = [0] * 10

        return predictions_id

    def plot_user_preferences(self, genre_list, cat_list):
        line = str(genre_list).replace("'", '').replace('[', '').replace(']', '').replace(' ', '')
        pd.Series(line.split(',')).value_counts().plot(kind='bar')
        line = str(cat_list).replace("'", '').replace('[', '').replace(']', '').replace(' ', '')
        pd.Series(line.split(',')).value_counts().plot(kind='bar')

    def recommend_for_user_set(self, test, column_name, is_plot):
        count_matrix, cosine_sim = self.create_count_matrix(column_name)
        total = len(test)
        recommendations = []
        for index, user in test.iterrows():
            print('{0}/{1}'.format(index, total))
            user_id = user['User_ID']
            predictions = self.get_recommendations_for_a_user(user_id, is_plot, cosine_sim)
            recommendations.append(predictions)

        return recommendations

    def user_analysis(self, user_id):
        user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == user_id])
        game_ids = user_info['Game_ID'].tolist()
        game_names = user_info['Game_Name'].tolist()
        game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
        cat_list = game_info['categories'].tolist()
        genre_list = game_info['genres'].tolist()


        user_info = pd.merge(user_info, self.dfs.game_ratings, on ='Game_ID')
        user_info['user_rating'] = user_info['Hours']/user_info['average_hours']
        user_info['user_rating_2'] = user_info['Hours']/user_info['Hours'].mean()
        # user_rating ve user_rating_2 paralel
        user_info['user_rating_3'] = user_info['user_rating']*user_info['user_rating_2']
        user_info.sort_values(by=['user_rating'])

        #sns.jointplot(x='user_rating', y='user_rating_3', data=user_info)
        #plt.show()

        # for genre
        self.plot_word_cloud(str(genre_list).replace("'", '').replace('[', '').replace(']', '').split(','))

        # for categories
        self.plot_word_cloud(str(cat_list).replace("'", '').replace('[', '').replace(']', '').split(','))

        print(user_info.head(10))

    def plot_word_cloud(self, data):
        wordcloud = WordCloud(background_color="white", stopwords=set(STOPWORDS))
        word_could_dict = Counter(data)
        wordcloud.generate_from_frequencies(word_could_dict)
        plt.figure(figsize=(5, 3))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def KNN(self):
        games = self.dfs.game_ratings.copy()
        games.drop('Game_Name', inplace = True, axis = 1)

        min_max_scaler = MinMaxScaler()
        games_features = min_max_scaler.fit_transform(games)
        np.round(games_features, 2)
        nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(games_features)
        distances, indices = nbrs.kneighbors(games_features)
        return distances, indices

    def get_index_from_name(self, name):
        index = self.dfs.game_ratings.index[self.dfs.game_ratings['Game_Name'] == str(name).lower()].tolist()
        return index

    def get_recommendation_with_knn(self, name):
        self.dfs.game_ratings = self.dfs.game_ratings.reset_index(drop = True)

        distances, indices = self.KNN()

        # query_index = self.get_index_from_name('mass effect')
        query_index = np.random.choice(self.dfs.game_ratings.shape[0])
        print('for {0}'.format(self.dfs.game_ratings['Game_Name'][query_index]))
        for query_index in indices[query_index][1:]:
            print(self.dfs.game_ratings['Game_Name'][query_index])

    def get_recommend_for_user_with_knn(self, user_id):
        user_info = pd.DataFrame(self.users_games.loc[self.users_games['User_ID'] == user_id])
        game_names = user_info['Game_Name'].tolist()
        game_info = self.played_games_info.loc[self.played_games_info['Game_Name'].isin(game_names)]
        return game_info

    def KNN_2(self, tfidf_matrix, predict_matrix):
        NUMBER_OF_CLUSTERS = 10
        km = KMeans(
            n_clusters=NUMBER_OF_CLUSTERS,
            init='k-means++',
            max_iter=500)
        km.fit(tfidf_matrix)

        clusters = km.predict(predict_matrix)

        return km, clusters

    def KNN_predict_2(self, km, matrix):

        # First: for each we get its corresponding cluster
        clusters = km.predict(matrix)

        return clusters

    def PCA(self, tfidf_matrix):
        # We train the PCA on the dense version of the tf-idf.
        pca = PCA(n_components=3)
        two_dim = pca.fit_transform(tfidf_matrix.todense())

        scatter_x = two_dim[:, 0]  # first principle component
        scatter_y = two_dim[:, 1]  # second principle component
        scatter_z = two_dim[:, 2]  # second principle component

        return scatter_x, scatter_y, scatter_z

    def plot_tfidf_matrix(self, clusters, km, scatter_x, scatter_y, scatter_z):
        plt.style.use('ggplot')

        fig = plt.figure(figsize=(5.5, 3))
        ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)

        #fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        # color map for NUMBER_OF_CLUSTERS we have
        labels_color_map = {
            0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
            5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
        }

        # group by clusters and scatter plot every cluster
        # with a colour and a label
        for group in np.unique(clusters):
            ix = np.where(clusters == group)
            ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=labels_color_map[group], label=group)

        centroids = km.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=160, c='red')

        ax.legend()
        plt.xlabel("PCA 0")
        plt.ylabel("PCA 1")
        plt.show()

    def print_top10words_in_clusters(self, km, features):
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        for i in range(10):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % features[ind], end='')
            print()

    def KNN_3(self, list1, list2, column_name):
        vectorizer = TfidfVectorizer(min_df=0, max_df=0.5, stop_words="english", ngram_range=(1, 3))
        vec = vectorizer.fit(list1[column_name])  # train vec using list1
        vectorized = vec.transform(list1[column_name])  # transform list1 using vec

        km = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, precompute_distances=True,
                    verbose=0, random_state=None, n_jobs=1)
        features = vectorizer.get_feature_names()
        km.fit(vectorized)
        list2Vec = vec.transform(list2[column_name])  # transform list2 using vec
        clusters = km.predict(list2Vec)

        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        for i in range(10):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % features[ind], end='')
            print()












import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from nltk.internals import Counter
from pandas.core import frame
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KNeighborsClassifier
from src.ContentBasedRecommendation import ContentBasedRecommendation
from src.Dataframes import Dataframes
from src.Evalution import Evaluation
from src.PopularGamesRecommendation import PopularGamesRecommendation
from src.RandomGamesRecommendation import RandomGamesRecommendation
from src.SimilarGamesRecommendation import SimilarGamesRecommendation

def EvaluatePredictions():
    prediction_list = []
    prediction_list.append(popularity_predictions)
    prediction_list.append(random_predictions)
    prediction_list.append(similarity_predictions)
    prediction_list.append(content_based_predictions)

    names = ['Popularity Recommender', 'Random Recommender', 'Similarity Recommender', 'Content-based Recommender']

    print('Mark metric evaluation...')
    evaluation.MarkMetricEvaluation(actual, prediction_list, names)
    print('Personalization evaluation...')
    evaluation.PersonalizationMetricEvaluation(prediction_list, names)
    print('Coverage metric evaluation...')
    catalog = dataframes.users_games.Game_ID.unique().tolist()
    evaluation.CoverageMetricEvaluation(catalog, prediction_list, names)


pd.set_option('max_columns', None)

evaluation = Evaluation()
dataframes = Dataframes.get_instance()

popular_rec = PopularGamesRecommendation()
random_rec = RandomGamesRecommendation()
content_based_rec = ContentBasedRecommendation()
similar_rec = SimilarGamesRecommendation()

# user_id = int(input('Enter a user id?'))
# print(type(user_id))

# dataframes.analyze_datasets()

# evaluation part #
#temp = dataframes.users_games.copy()
#temp = temp.groupby('User_ID').filter(lambda x : len(x) > 10)
#temp.drop("Hours", inplace=True, axis=1)
#temp.drop("Game_Name", inplace=True, axis=1)
#print(temp.groupby('User_ID').get_group(115))
#
#test = pd.DataFrame()
#test['User_ID'] = temp.User_ID.unique()
#grouped = temp.groupby('User_ID')
#
#games = []
#for index, row in test.iterrows():
#    user_id = row['User_ID']
#    games.append(grouped.get_group(user_id)['Game_ID'].values.tolist())
#
#test['actual'] = games
#print(len(test['actual']))
#print(test.head(10))
#print()
#
#print('...popular games recommendation process...')
#test['popularity_predictions'] = popular_rec.recommend_for_user_set(test)
#print(test['popularity_predictions'])
#print()
#
#print('...similar games recommendation process...')
#test['similarity_predictions'] = similar_rec.recommend_for_user_set(test, 'Portal 2')
#print(test['similarity_predictions'])
#print()
#
#print('...random games recommendation process...')
#test['random_predictions'] = random_rec.recommend_for_user_set(test)
#print(test['random_predictions'])
#print()
#
#print('...content based games recommendation process...')
#test['content_based_predictions'] = content_based_rec.recommend_for_user_set(test, 'short_description', is_plot=False)
#print(test['content_based_predictions'])
#print()
#
#print('All recommendation processes were over.')
#
#print(test.head(10))
#actual = test.actual.values.tolist()
#similarity_predictions = test.similarity_predictions.values.tolist()
#popularity_predictions = test.popularity_predictions.values.tolist()
#random_predictions = test.random_predictions.values.tolist()
#content_based_predictions = test.content_based_predictions.values.tolist()
#
#print('Evaluating process is starting...')
#EvaluatePredictions()

# end of evaluation part #

content_based_rec.user_analysis(1)
#content_based_rec.get_similar_games(name='spore')
#content_based_rec.get_recommendations_with_tfidf('short_description', 'fallout 4')
#content_based_rec.get_recommendations_with_count_matrix('short_description', 'fallout 4')
#content_based_rec.get_recommendation_with_knn('portal 2')

#cosine_sim, tfidf_matrix, features = content_based_rec.create_tfidf_matrix('genres', dataframes.played_games_info)
#print(tfidf_matrix.shape)
#games = content_based_rec.get_recommend_for_user_with_knn(1)
#cosine_sim2, tfidf_matrix2, features2 = content_based_rec.create_tfidf_matrix('genres', games)
#print(tfidf_matrix2.shape)
#km, clusters = content_based_rec.KNN_2(tfidf_matrix, tfidf_matrix2)
#scatter_x, scatter_y, scatter_z = content_based_rec.PCA(tfidf_matrix)
#content_based_rec.plot_tfidf_matrix(clusters, km, scatter_x, scatter_y, scatter_z)

#km, clusters = content_based_rec.KNN_2(tfidf_matrix)
#scatter_x, scatter_y, scatter_z = content_based_rec.PCA(tfidf_matrix)
#content_based_rec.plot_tfidf_matrix(clusters, km, scatter_x, scatter_y, scatter_z)

#games = content_based_rec.get_recommend_for_user_with_knn(1)
#content_based_rec.KNN_3(dataframes.played_games_info, games, 'genres')

plt.show()




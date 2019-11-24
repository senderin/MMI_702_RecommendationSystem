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
from src.Data import Data
from src.Evalution import Evaluation
from src.PopularGamesRecommendation import PopularGamesRecommendation
from src.RandomGamesRecommendation import RandomGamesRecommendation
from src.SimilarGamesRecommendation import SimilarGamesRecommendation
from src.User_Analysis import User_Analysis


def precision_recall(actual, prediction):
    precision = evaluation.precision(actual, prediction)
    recall = evaluation.recall(actual, prediction)
    print('Precision: {0}'.format(precision))
    print('Recall: {0}'.format(recall))
    if not recall == 0:
        print('F1 Score: {0}'.format(evaluation.f1_score(precision, recall)))
    return precision, recall

def evaluate_predictions():

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
    catalog = data.users_games.Game_ID.unique().tolist()
    evaluation.CoverageMetricEvaluation(catalog, prediction_list, names)

def recommend_for_all_users(type_name):
    temp = data.users_games.copy()
    temp = temp.groupby('User_ID').filter(lambda x : len(x) > 10)
    temp.drop("Hours", inplace=True, axis=1)
    temp.drop("Game_Name", inplace=True, axis=1)
    print(temp.groupby('User_ID').get_group(115))

    test = pd.DataFrame()
    test['User_ID'] = temp.User_ID.unique()
    grouped = temp.groupby('User_ID')

    games = []
    for index, row in test.iterrows():
       user_id = row['User_ID']
       games.append(grouped.get_group(user_id)['Game_ID'].values.tolist())

    test['actual'] = games
    print(len(test['actual']))
    print(test.head(10))
    print()

    print('...popular games recommendation process...')
    test['popularity_predictions'] = popular_rec.recommend_for_user_set(test)
    print(test['popularity_predictions'])
    print()

    print('...similar games recommendation process...')
    test['similarity_predictions'] = similar_rec.recommend_for_user_set(test, 'Portal 2')
    print(test['similarity_predictions'])
    print()

    print('...random games recommendation process...')
    test['random_predictions'] = random_rec.recommend_for_user_set(test)
    print(test['random_predictions'])
    print()

    print('...content based games recommendation process...')
    test['content_based_predictions'] = content_based_rec.recommend_for_user_set(test, type_name, is_plot=False)
    print(test['content_based_predictions'])
    print()

    print('All recommendation processes were over.')

    print(test.head(10))
    actual = test.actual.values.tolist()
    similarity_predictions = test.similarity_predictions.values.tolist()
    popularity_predictions = test.popularity_predictions.values.tolist()
    random_predictions = test.random_predictions.values.tolist()
    content_based_predictions = test.content_based_predictions.values.tolist()

    print('Evaluating process is starting...')
    evaluate_predictions()

def ask_metadata():
    print('1. Name')
    print('2. Short description')
    print('3. Detailed description')
    print('4. About the game')
    type = int(input('Enter the metadata number to be used in content-based rec. model: '))
    type_name = ''
    if type == 1:
        type_name = 'Game_Name'
    elif type == 2:
        type_name = 'short_description'
    elif type == 3:
        type_name = 'detailed_description'
    elif type == 4:
        type_name = 'about_the_game'
    return type_name


pd.set_option('max_columns', None)

evaluation = Evaluation()
data = Data.get_instance()

popular_rec = PopularGamesRecommendation()
random_rec = RandomGamesRecommendation()
content_based_rec = ContentBasedRecommendation()
similar_rec = SimilarGamesRecommendation()

print('1. Get recommendation for a game')
print('2. Get recommendation for a player')
print('3. Get recommendation for all players')
choose = int(input('Enter your choose number: '))

if choose == 1:
    game_name = input('Enter name of the game: ')
    metadata_name = ask_metadata()
    cb_names = content_based_rec.recommend_for_game(game_name, metadata_name)
    names = similar_rec.recommend_for_game(game_name)

    temp = pd.DataFrame()
    temp['CB Rec. List'] = cb_names
    temp['Sim. Rec. List'] = names
    print(temp.head(10))

elif choose == 2:
    user_id = np.random.choice(data.users_games['User_ID'].unique())
    user_analysis = User_Analysis(user_id)
    user_analysis.user_profile_description()

    played_games = user_analysis.get_most_played_N_games(1)

    print('Recommendation for User ID {0}:'.format(user_id))
    r_ids, r_names = random_rec.recommend_for_user()
    p_ids, p_names = popular_rec.recommend_for_user()
    cb_ids_1, cb_names_1 = content_based_rec.recommend_for_user(played_games, 'short_description')
    cb_ids_2, cb_names_2 = content_based_rec.recommend_for_user(played_games, 'detailed_description')
    cb_ids_3, cb_names_3 = content_based_rec.recommend_for_user(played_games, 'about_the_game')
    cb_ids_4, cb_names_4 = content_based_rec.recommend_with_all_metadata(played_games)

    temp = pd.DataFrame()
    temp['CB Rec. List (short_description)'] = cb_names_1
    print(len(temp['CB Rec. List (short_description)']))
    temp['CB Rec. List (detailed_description)'] = cb_names_2
    temp['CB Rec. List (about_the_game)'] = cb_names_3
    temp['CB Rec. List (user_analysis)'] = cb_names_4
    temp['Random Rec. List'] = r_names
    temp['Popular Rec. List'] = p_names
    print(temp.head(10))

    precisions = []
    recalls = []
    actual = user_analysis.get_actual_list()

    print('For random rec. model: ')
    p, r = precision_recall(actual, r_ids)
    precisions.append(p)
    recalls.append(r)

    print('For popular rec. model: ')
    p, r = precision_recall(actual, p_ids)
    precisions.append(p)
    recalls.append(r)

    print('For cb rec. model (short_description): ')
    p, r = precision_recall(actual, cb_ids_1)
    precisions.append(p)
    recalls.append(r)

    print('For cb rec. model (detailed_description): ')
    p, r = precision_recall(actual, cb_ids_2)
    precisions.append(p)
    recalls.append(r)

    print('For cb rec. model (about_the_game): ')
    p, r = precision_recall(actual, cb_ids_3)
    precisions.append(p)
    recalls.append(r)

    print('For cb rec. model (all metadata): ')
    p, r = precision_recall(actual, cb_ids_4
                            )
    precisions.append(p)
    recalls.append(r)

    evaluation.plot_precision_recall(precisions, recalls)

elif choose == 3:
    metadata_name = ask_metadata()
    recommend_for_all_users(metadata_name)

else:
    print('Invalid number!')







#content_based_rec.user_analysis(1)
#content_based_rec.get_similar_games(name='spore')
#content_based_rec.get_recommendations_with_tfidf('short_description', 'fallout 4')
#content_based_rec.get_recommendations_with_count_matrix('short_description', 'fallout 4')
#content_based_rec.get_recommendation_with_knn('portal 2')




#results = content_based_rec.KNN_4(tfidf_matrix, 13)
#print(results)
#print(tfidf_matrix.shape)
#games = content_based_rec.get_recommend_for_user_with_knn(1)
#cosine_sim2, tfidf_matrix2, features2 = content_based_rec.create_tfidf_matrix('genres', games)
#print(tfidf_matrix2.shape)
#km, clusters = content_based_rec.KNN_2(tfidf_matrix, tfidf_matrix2)
#scatter_x, scatter_y, scatter_z = content_based_rec.PCA(tfidf_matrix)
#content_based_rec.plot_tfidf_matrix(clusters, km, scatter_x, scatter_y, scatter_z)
#
#km, clusters = content_based_rec.KNN_2(tfidf_matrix)
#scatter_x, scatter_y, scatter_z = content_based_rec.PCA(tfidf_matrix)
#content_based_rec.plot_tfidf_matrix(clusters, km, scatter_x, scatter_y, scatter_z)
#
#games = content_based_rec.get_recommend_for_user_with_knn(1)
#content_based_rec.KNN_3(data.played_games_info, games, 'genres')
#content_based_rec.recommend_for_game("portal 2", 'Game_Name')

plt.show()




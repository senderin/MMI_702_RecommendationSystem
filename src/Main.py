import random
import numpy
import pandas as pd
from numpy import mean
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.ContentBasedRecommendation import ContentBasedRecommendation
from src.Data import Data
from src.Evalution import Evaluation
from src.user_analysis import User_Analysis

def get_rec_for(game_name, metadata_name):
    cb_ids = []
    cb_names = []
    if not metadata_name is 'all_metadata':
        cb_ids, cb_names = content_based_rec.recommend_for_game(game_name, metadata_name)

    else:
        played_games = data.played_games.loc[data.played_games['Game_Name'] == game_name]
        cb_ids, cb_names = content_based_rec.recommend_with_all_metadata(played_games)

    return cb_ids ,cb_names

def rec_for_game(game_name):

    cb_ids_name, cb_names_name = get_rec_for(game_name, 'Game_Name')
    cb_ids_short_desc, cb_names_short_desc = get_rec_for(game_name, 'short_description')
    cb_ids_detailed_desc, cb_names_detailed_desc = get_rec_for(game_name, 'detailed_description')
    cb_ids_about, cb_names_about = get_rec_for(game_name, 'about_the_game')
    cb_ids_all, cb_names_all = get_rec_for(game_name, 'all_metadata')

    temp = pd.DataFrame()
    temp['Name_Game_Name'] = cb_names_name
    temp['Name_short_description'] = cb_names_short_desc
    temp['Name_detailed_description'] = cb_names_detailed_desc
    temp['Name_about_the_game'] = cb_names_about
    temp['Name_all_metadata'] = cb_names_all
    temp['Ids_Game_Name'] = cb_ids_name
    temp['Ids_short_description'] = cb_ids_short_desc
    temp['Ids_detailed_description'] = cb_ids_detailed_desc
    temp['Ids_about_the_game'] = cb_ids_about
    temp['Ids_all_metadata'] = cb_ids_all
    #print(temp_name.head(10))
    #print()

    return temp

pd.set_option('max_columns', None)

evaluation = Evaluation()
data = Data.get_instance()
content_based_rec = ContentBasedRecommendation()

#user_id = 1
user_id = numpy.random.choice(data.users_games['User_ID'].unique())
print("RECOMMENDATION FOR USER ID-{0}".format(user_id))
user_analysis = User_Analysis(user_id)
user_analysis.user_profile_description()

most_played_3_games = user_analysis.get_most_played_N_games(3)
pool = pd.DataFrame()
for index, row in most_played_3_games.iterrows():
    print("Game Name: {0}".format(row['Game_Name']))
    if len(pool) == 0:
        pool = rec_for_game(row['Game_Name']).head(4)
    else:
        pool = pool.append(rec_for_game(row['Game_Name']).head(3))
print("Size of the pool: {0}".format(len(pool)))

predictions = pd.DataFrame()
if not len(pool) == 0:
    predictions = pool.sample(10);
else:
    predictions = [0] * 10
print("Recommendation of the Models:")
print(predictions.head(10))

precisions = []
recalls = []
actual = user_analysis.get_actual_id_list()

print('For cb rec. model (game_name): ')
p, r = evaluation.precision_recall(actual, predictions['Ids_Game_Name'])
precisions.append(p)
recalls.append(r)
print()

print('For cb rec. model (short_description): ')
p, r = evaluation.precision_recall(actual, predictions['Ids_short_description'])
precisions.append(p)
recalls.append(r)
print()

print('For cb rec. model (detailed_description): ')
p, r = evaluation.precision_recall(actual, predictions['Ids_detailed_description'])
precisions.append(p)
recalls.append(r)
print()

print('For cb rec. model (about_the_game): ')
p, r = evaluation.precision_recall(actual, predictions['Ids_about_the_game'])
precisions.append(p)
recalls.append(r)
print()

print('For cb rec. model (all metadata): ')
p, r = evaluation.precision_recall(actual, predictions['Ids_all_metadata'])

precisions.append(p)
recalls.append(r)

titles = ('(Game_Name)', '(short_description)', '(detailed_description)', '(about_the_game)','(all_metadata)')
evaluation.plot_precision_recall_bar_chart(precisions, recalls, titles)

# plotting genre and categories distribution

def plot_genre_cat_distribution(id_column_name, axes, x, y, title):
    predictions_info = pd.DataFrame()
    predictions_info = data.played_games.loc[data.played_games['Game_ID'].isin(predictions[id_column_name].values.tolist())]

    #cat_list = predictions_info['categories'].tolist()
    genre_list = predictions_info['genres'].tolist()
    genres = str(genre_list).replace("'", '').replace('[', '').replace(']', '').replace(',', ';').replace(' ', '').split(';')
    #categories = str(cat_list).replace("'", '').replace('[', '').replace(']', '').replace(',', ';').replace(' ','').split(';')

    pd.Series(genres).value_counts(normalize=True, sort=True, ascending=False)[:10].plot(ax=axes[x,y], kind='bar', title =title, fontsize = 8)
    #pd.Series(categories).value_counts(sort=True, ascending=False).plot(kind='bar')

fig, axes = plt.subplots(nrows=2, ncols=3)

user_analysis.plot_genre_cat_preferences(axes, 0, 0)
plot_genre_cat_distribution('Ids_Game_Name', axes, 0, 1, 'Game_Name')
plot_genre_cat_distribution('Ids_short_description', axes, 0, 2, 'short_description')
plot_genre_cat_distribution('Ids_detailed_description', axes, 1, 0, 'detailed_description')
plot_genre_cat_distribution('Ids_about_the_game', axes, 1, 1, 'about_the_game')
plot_genre_cat_distribution('Ids_all_metadata', axes, 1, 2, 'all_metadata')
plt.subplots_adjust(hspace=1)
plt.show()






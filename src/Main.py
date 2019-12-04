import pandas as pd
from numpy import mean
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.ContentBasedRecommendation import ContentBasedRecommendation
from src.Data import Data
from src.Evalution import Evaluation

def get_rec_for(game_name, metadata_name):
    cb_ids = []
    cb_names = []
    if not metadata_name is 'all_metadata':
        cb_ids, cb_names = content_based_rec.recommend_for_game(game_name, metadata_name)

    else:
        played_games = data.played_games.loc[data.played_games['Game_Name'] == game_name]
        cb_ids, cb_names = content_based_rec.recommend_with_all_metadata(played_games)

    return cb_ids ,cb_names

pd.set_option('max_columns', None)

evaluation = Evaluation()
data = Data.get_instance()
content_based_rec = ContentBasedRecommendation()

def rec_for_game(game_name):

    cb_ids_name, cb_names_name = get_rec_for(game_name, 'Game_Name')
    cb_ids_short_desc, cb_names_short_desc = get_rec_for(game_name, 'short_description')
    cb_ids_detailed_desc, cb_names_detailed_desc = get_rec_for(game_name, 'detailed_description')
    cb_ids_about, cb_names_about = get_rec_for(game_name, 'about_the_game')
    cb_ids_all, cb_names_all = get_rec_for(game_name, 'all_metadata')

    temp = pd.DataFrame()
    temp['CB Rec. List (Game_Name)'] = cb_names_name
    temp['CB Rec. List (short_description)'] = cb_names_short_desc
    temp['CB Rec. List (detailed_description)'] = cb_names_detailed_desc
    temp['CB Rec. List (about_the_game)'] = cb_names_about
    temp['CB Rec. List (all_metadata)'] = cb_names_all
    print(temp.head(10))
    print()

    # EVALUATION
    print('\nEvaluation of CB Rec. List (Game_Name)')
    p_name, r_name = evaluation.evaluate_recommendations(game_name, cb_ids_name, cb_names_name, False)

    print('\nEvaluation of CB Rec. List (short_description)')
    p_short_desc, r_short_desc = evaluation.evaluate_recommendations(game_name, cb_ids_short_desc, cb_names_short_desc, False)

    print('\nEvaluation of CB Rec. List (detailed_description)')
    p_detailed_desc, r_detailed_desc = evaluation.evaluate_recommendations(game_name, cb_ids_detailed_desc, cb_names_detailed_desc, False)

    print('\nEvaluation of CB Rec. List (about_the_game)')
    p_about, r_about = evaluation.evaluate_recommendations(game_name, cb_ids_about, cb_names_about, False)

    print('\nEvaluation of CB Rec. List (all_metadata)')
    p_all, r_all = evaluation.evaluate_recommendations(game_name, cb_ids_all, cb_names_all, False)

    # plot
    precisions = (p_name, p_short_desc, p_detailed_desc, p_about, p_all)
    recalls = (r_name, r_short_desc, r_detailed_desc, r_about, r_all)

    titles = ('(Game_Name)', '(short_description)', '(detailed_description)', '(about_the_game)','(all_metadata)')
    evaluation.plot_precision_recall_bar_chart(precisions, recalls, titles)

    return precisions, recalls

game_name = input('Enter name of the game: ')
rec_for_game(game_name)

#random_games = data.played_games.Game_Name.sample(150).values.tolist()
#print(random_games)
#precisions = []
#recalls = []
#for i in range(len(random_games)):
#    precision, recall = rec_for_game(random_games[i])
#    precisions.append(precision)
#    recalls.append(recall)
#
#p_name  = mean([x[0] for x in precisions])
#p_short_desc = mean([x[1] for x in precisions])
#p_detailed = mean([x[2] for x in precisions])
#p_about = mean([x[3] for x in precisions])
#p_all = mean([x[4] for x in precisions])
#
#print('Average Precision (Game Name) : {0}'.format(p_name))
#print('Average Precision (Short Description) : {0}'.format(p_short_desc))
#print('Average Precision (Detailed Description) : {0}'.format(p_detailed))
#print('Average Precision (About the Game) : {0}'.format(p_about))
#print('Average Precision (All Metadata) : {0}'.format(p_all))
#
#r_name  = mean([x[0] for x in recalls])
#r_short_desc = mean([x[1] for x in recalls])
#r_detailed = mean([x[2] for x in recalls])
#r_about = mean([x[3] for x in recalls])
#r_all = mean([x[4] for x in recalls])
#
#print('Average Recall (Game Name) : {0}'.format(r_name))
#print('Average Recall (Short Description) : {0}'.format(r_short_desc))
#print('Average Recall (Detailed Description) : {0}'.format(r_detailed))
#print('Average Recall (About the Game) : {0}'.format(r_about))
#print('Average Recall (All Metadata) : {0}'.format(r_all))
#
#titles = ('(Game_Name)', '(short_description)', '(detailed_description)', '(about_the_game)','(all_metadata)')
#precisions = (p_name, p_short_desc, p_detailed, p_about, p_all)
#recalls = (r_name, r_short_desc, r_detailed, r_about, r_all)
#evaluation.plot_precision_recall_bar_chart(precisions, recalls, titles)







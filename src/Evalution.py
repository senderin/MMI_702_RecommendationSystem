import numpy
import numpy as np
import pandas
import recmetrics
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd

from src.Data import Data


class Evaluation():

    # mar@k evaluation metric
    # how well the recommender is able to recall all the items the user has rated positively in the test set
    def MarkMetric(self, actual, predictions):
        mark = []
        for K in np.arange(1, 11):
            mark.extend([recmetrics.mark(actual, predictions, k=K)])

        return mark

    def PlotMarkMetricResult(self, mark_scores, names):
        index = range(1, 10 + 1)
        fig = plt.figure(figsize=(15, 7))
        recmetrics.mark_plot(mark_scores, model_names=names, k_range=index)
        plt.show()

    def MarkMetricEvaluation(self, actual, prediction_list, names):
        mark_scores = []
        for index in range(len(prediction_list)):
            mark_scores.append(self.MarkMetric(actual, prediction_list[index]))

        print('Mark scores: {0}'.format(mark_scores))
        self.PlotMarkMetricResult(mark_scores, names)

    # the dissimilarity between user's lists of recommendations
    # a low personalization score indicates user's recommendations are very similar
    def PersonalizationMetric(self, predictions):
        personalization_value = recmetrics.personalization(predictions)
        return personalization_value

    def PersonalizationMetricEvaluation(self, prediction_list, names):
        personalization_values = []
        for index in range(len(prediction_list)):
            personalization_values.append(self.PersonalizationMetric(prediction_list[index]))

        print('Personalization scores: {0}'.format(personalization_values))
        self.PlotPersonalizationMetricResult(personalization_values, names)

    def PlotPersonalizationMetricResult(self, personalization_values, names):
        index = np.arange(len(names))
        plt.bar(index, personalization_values)
        plt.xlabel('Models', fontsize=5)
        plt.ylabel('Personalization', fontsize=5)
        plt.xticks(index, names, fontsize=5, rotation=30)
        plt.show()

    # the cosine similarity between the items in a list of recommendations
    def IntralistMetric(self, predictions, feature_dataframe):
        value = recmetrics.intra_list_similarity(predictions, feature_dataframe)
        return value

    def PlotIntralistMetricResult(self, intralist_values, names):
        index = np.arange(len(names))
        plt.bar(index, intralist_values)
        plt.xlabel('Models', fontsize=5)
        plt.ylabel('Intra-list', fontsize=5)
        plt.xticks(index, names, fontsize=5, rotation=30)
        plt.show()

    def IntralistMetricEvaluation(self, feature_dataframe, prediction_list, names):
        intralist_values = []
        for index in range(len(prediction_list)):
            intralist_values.append(self.IntralistMetric(prediction_list[index], feature_dataframe))

        self.PlotIntralistMetricResult(intralist_values, names)

    # the percent of items that the recommender is able to recommend
    def CoverageMetric(self, predictions, catalog):
        coverage_value = recmetrics.coverage(predictions, catalog)
        return coverage_value

    def PlotCoverageMetricResult(self, coverage_scores, names):
        fig = plt.figure(figsize=(15, 7))
        recmetrics.coverage_plot(coverage_scores, names)
        plt.show()

    def CoverageMetricEvaluation(self, catalog, prediction_list, names):
        coverage_scores = []
        for index in range(len(prediction_list)):
            coverage_scores.append(self.CoverageMetric(prediction_list[index], catalog))
        print('coverage scores: {0}'.format(coverage_scores))
        self.PlotCoverageMetricResult(coverage_scores, names)

    def precision_for_predictions(self, actual, predictions):
        precisions = []
        for index in range(len(predictions)):
            intersect = len(self.intersection(predictions[index], actual[index]))
            precision = intersect / len(predictions[index])
            precisions.append(precision)
            if precision == 0:
                print('predictions {0}'.format(predictions[index]))
                print('actuals {0}'.format(actual[index]))
                print("intersect {0} - len(actual[index]) {1}".format(intersect, len(predictions[index])))
                print()

        return precisions, sum(precisions) / len(precisions)

    def precision(self, actual, prediction):
        intersect = len(self.intersection(prediction, actual))
        precision = intersect / len(prediction)
        return precision

    def recall_for_predictions(self, actual, predictions):
        recalls = []
        for index in range(len(predictions)):
            intersect = len(self.intersection(predictions[index], actual[index]))
            # '- 1' presents the game that is used for recommendation
            recall = intersect / (len(actual) - 1)
            recalls.append(recall)
        return recalls, sum(recalls) / len(recalls)

    def recall(self, actual, prediction):
        intersect = len(self.intersection(prediction, actual))
        recall = intersect / len(actual)
        return recall

    def plot_precision_recall(self, precisions, recalls):
        plt.style.use('seaborn-whitegrid')
        plt.figure()
        plt.subplot(211)
        plt.plot(precisions, color='tab:blue')

        plt.subplot(212)
        plt.plot(recalls, color='tab:orange')
        plt.show()

    def intersection(self, lst1, lst2):
        return list(set(lst1) & set(lst2))

    # weighted average of precision an recall
    def f1_score(self, precision, recall):
        return 0.2 * (precision * recall) / (precision + recall)

    def evaluate_recommendations(self, game_name, rec_list_id, rec_list_name, show_all):
        evaluation = Evaluation()
        users_played_X = self.find_users_played_X(game_name)
        print('# of players that play the game: {0}'.format(len(users_played_X)))
        precisions = 0
        recalls = 0
        if users_played_X.shape[0] > 0:
            for index, row in users_played_X.iterrows():
                intersect = len(evaluation.intersection(rec_list_name, row['Game_Name']))
                precision = evaluation.precision(row['Game_ID'], rec_list_id)
                precisions = precisions + precision
                recall = evaluation.recall(row['Game_ID'], rec_list_id)
                recalls = recalls + recall
                if show_all:
                    print(row['Game_Name'])
                    print('I: {2} P: {0} R: {1}'.format(precision, recall, intersect))

        average_precision = precisions/len(users_played_X)
        average_recall = recalls/len(users_played_X)
        print('Average P: {0} Average R: {1}'.format(average_precision, average_recall))
        if average_recall > 0:
            f1_score = self.f1_score(average_precision, average_recall)
            print('Average F1 score: {0}'.format(f1_score))
        return average_precision, average_recall


    def find_users_played_X(self, game_name):
        data = Data.get_instance()
        game_id = data.played_games.loc[data.played_games['Game_Name'] == game_name]['Game_ID']
        # print(game_id)

        temp = data.users_games.copy()
        temp = temp.groupby('User_ID').filter(lambda x: len(x) > 2)
        temp.drop("Hours", inplace=True, axis=1)

        grouped = temp.groupby('User_ID')

        users = pd.DataFrame()
        users['User_ID'] = data.users_games['User_ID'].unique()

        id_list = []
        name_list = []
        for index, row in users.iterrows():
            user_id = row['User_ID']
            ids = grouped.get_group(user_id)['Game_ID'].values.tolist()
            names = grouped.get_group(user_id)['Game_Name'].values.tolist()
            # print(names)
            id_list.append(ids)
            name_list.append(names)

        users['Game_Name'] = name_list
        users['Game_ID'] = id_list
        # print(users.head(10))

        ids = [game_id]
        users_played_X = pd.DataFrame()
        for index, row in users.iterrows():
            if any(game_name in s for s in row.Game_Name):
                users_played_X = users_played_X.append(row, ignore_index=True)

        # print(users_played_X.head(10))
        return users_played_X

    def plot_precision_recall_bar_chart(self, precisions, recalls, titles):
        # create plot
        fig, ax = plt.subplots()
        n_groups = len(precisions)
        index = numpy.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, precisions, bar_width,
                         alpha=opacity,
                         color='b',
                         label='precision')

        rects2 = plt.bar(index + bar_width, recalls, bar_width,
                         alpha=opacity,
                         color='g',
                         label='recall')

        plt.xlabel('Rec. Models')
        plt.ylabel('Scores')
        plt.title('Precision-Recall of Rec. Models')
        plt.xticks(index + bar_width, (titles), rotation=90)
        plt.legend()

        plt.tight_layout()
        plt.show()








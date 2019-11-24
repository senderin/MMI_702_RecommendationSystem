import numpy as np
import pandas
import recmetrics
import matplotlib.pyplot as plt
from statistics import mean

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
            recall = intersect / len(actual)
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
        return 2 * (precision * recall) / (precision + recall)








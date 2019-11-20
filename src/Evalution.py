import numpy as np
import recmetrics
import matplotlib.pyplot as plt

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




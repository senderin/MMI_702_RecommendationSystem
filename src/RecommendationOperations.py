import pandas as pd
import matplotlib.pyplot as plt
import recmetrics as recmetrics
import seaborn as sns
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KNeighborsClassifier

def get_users_predictions(user_id, n, model):
    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()

pd.set_option('max_columns', None)

playtimes = pd.read_csv('playtime.csv', sep='\t')
games = pd.read_csv('game_id_name.csv', sep='\t')

print(playtimes.loc[playtimes['Game_ID'].isin(games['Game_ID'])].count())
#print(playtimes.head())
#print(games.head())
#print(playtimes.shape)
#print(games.shape)

merged_dataset = pd.merge(playtimes, games, on='Game_ID')
#print(merged_dataset.head())
#print(merged_dataset.shape)

# long tail example
#fig = plt.figure(figsize=(15, 7))
#recmetrics.long_tail_plot(df=merged_dataset,
#             item_id_column="Game_ID",
#             interaction_type="movie ratings",
#             percentage=0.5,
#             x_labels=False)

print(merged_dataset.describe())

user_ids = merged_dataset['User_ID'].unique()
game_ids = merged_dataset['Game_ID'].unique()
#print(len(user_ids))
#print(len(game_ids))

games_count = merged_dataset['User_ID'].value_counts()
#print(games_count.head())

#plt.hist(merged_dataset['Game_ID'], bins=60)
#plt.hist(merged_dataset['User_ID'], bins=60)


# prec. according popularity

# # calculate the average playtime for each game
mean = merged_dataset['Hours'].mean()
playtimes_count = pd.DataFrame(games)
playtimes_count['average_hours'] = pd.DataFrame(merged_dataset.groupby(['Game_ID'])['Hours'].mean())
# print(playtimes_count.head(5))
# calculate number of ratings for each movie
playtimes_count['number_of_player'] = merged_dataset.groupby(['Game_ID'])['Game_ID'].count()
playtimes_count['rating'] = playtimes_count['average_hours'] / mean

# eliminate played less than 100 person
playtimes_count = playtimes_count.loc[playtimes_count['number_of_player']>100]

print("*************************")
#print(playtimes_count.keys())
print(playtimes_count.head(5))
print(playtimes_count['rating'].min())
print(playtimes_count['rating'].max())


# visualize the relationship between the playtime of a game and the number of players
#sns.jointplot(x='Hours', y='number_of_player', data=playtimes_count)
print("*************************")
# most-played games
top10 = playtimes_count.sort_values('average_hours', ascending=False).head(10)
print(top10.head(10))
print("*************************")
# most-owned games
top10 = playtimes_count.sort_values('number_of_player', ascending=False).head(10)
print(top10.head(10))
print("*************************")
# most-liked games
top10 = playtimes_count.sort_values('rating', ascending=False).head(10)
print(top10.head(10))
print("*************************")

# simple recommendations
# user - item matrix
playtime_matrix = merged_dataset.pivot_table(index='User_ID', columns='Game_Name', values='Hours').fillna(0)
#print(playtime_matrix.head(3))
game_portal = playtime_matrix['Portal 2']
similar_to_portal = playtime_matrix.corrwith(game_portal)
#print(similar_to_portal.head(10))

#drop those null values
# and transform correlation results into dataframes to make the results look more appealing
corr_contact = pd.DataFrame(similar_to_portal, columns=['Correlation'])
corr_contact.dropna(inplace=True)
print(corr_contact.sort_values('Correlation', ascending=False).head(10))

# implementation of CF recommender system by using SVD algorithm #
#format data for surprise
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(merged_dataset[['User_ID', 'Game_ID', 'Hours']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

#train SVD recommender
algo = SVD()
algo.fit(trainset)

#make predictions on test set.
test = algo.test(testset)
test = pd.DataFrame(test)
test.drop("details", inplace=True, axis=1)
test.columns = ['User_ID', 'Game_ID', 'actual', 'cf_predictions']
print(test.head())

#evaluate model with MSE and RMSE
print(recmetrics.mse(test.actual, test.cf_predictions))
print(recmetrics.rmse(test.actual, test.cf_predictions))


#create model (matrix of predicted values)
cf_model = test.pivot_table(index='User_ID', columns='Game_ID', values='cf_predictions').fillna(0)

#get example prediction
get_users_predictions(2472, 550, cf_model)

# format test data
test = test.copy().groupby('User_ID')['Game_ID'].agg({'actual': (lambda x: list(sns.set(x)))})

# make recommendations for all members in the test data
recs = []
for user in test.index:
    cf_predictions = get_users_predictions(user, 10, cf_model)
    recs.append(cf_predictions)

test['cf_predictions'] = recs
print(test.head())

# popularity recommender #

# make recommendations for all members in the test data
popularity_recs = merged_dataset.Game_ID.value_counts().head(10).index.tolist()

recs = []
for user in test.index:
    pop_predictions = popularity_recs
    recs.append(pop_predictions)

test['pop_predictions'] = recs
print(test.head(10))

recs = []
corr_contact = pd.DataFrame(similar_to_portal, columns=['Correlation'])
corr_contact.dropna(inplace=True)
x = corr_contact.sort_values('Correlation', ascending=False).head(10)
x = x.merge(games, on='Game_Name')
print(x)
for user in test.index:
    first_rec = x['Game_ID'].values.tolist()
    recs.append(first_rec)

test['first_rec'] = recs
print(test.head(10))

# random recommender #

# make recommendations for all members in the test data

recs = []
for user in test.index:
    random_predictions = merged_dataset.Game_ID.sample(10).values.tolist()
    recs.append(random_predictions)

test['random_predictions'] = recs
print(test.head(10))

# recall

actual = test.actual.values.tolist()
first_predictions = test.first_rec.values.tolist()
cf_predictions = test.cf_predictions.values.tolist()
pop_predictions = test.pop_predictions.values.tolist()
random_predictions = test.random_predictions.values.tolist()

first_mark = []
for K in np.arange(1, 11):
    first_mark.extend([recmetrics.mark(actual, first_predictions, k=K)])
print(first_mark)

pop_mark = []
for K in np.arange(1, 11):
    pop_mark.extend([recmetrics.mark(actual, pop_predictions, k=K)])
print(pop_mark)

random_mark = []
for K in np.arange(1, 11):
    random_mark.extend([recmetrics.mark(actual, random_predictions, k=K)])
print(random_mark)

cf_mark = []
for K in np.arange(1, 11):
    cf_mark.extend([recmetrics.mark(actual, cf_predictions, k=K)])
cf_mark

mark_scores = [random_mark, pop_mark, cf_mark, first_mark]
index = range(1,10+1)
names = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter', 'First Rec']

#fig = plt.figure(figsize=(15, 7))
#recmetrics.mark_plot(mark_scores, model_names=names, k_range=index)

# coverage
catalog = merged_dataset.Game_ID.unique().tolist()
first_coverage = recmetrics.coverage(first_predictions, catalog)
random_coverage = recmetrics.coverage(random_predictions, catalog)
pop_coverage = recmetrics.coverage(pop_predictions, catalog)
cf_coverage = recmetrics.coverage(cf_predictions, catalog)

coverage_scores = [random_coverage, pop_coverage, cf_coverage, first_coverage]
model_names = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter', 'First Rec']

#fig = plt.figure(figsize=(7, 5))
#recmetrics.coverage_plot(coverage_scores, model_names)

# personalization

print("**************************")
#print(recmetrics.personalization(pop_predictions))
#print(recmetrics.personalization(random_predictions))
#print(recmetrics.personalization(cf_predictions))
#print(recmetrics.personalization(first_predictions))
print("**************************")

#feature_df = movies[['Action', 'Comedy', 'Romance']]
#print(recmetrics.intra_list_similarity(cf_predictions, feature_df))
#print(recmetrics.intra_list_similarity(random_predictions, feature_df))
#print(recmetrics.intra_list_similarity(pop_predictions, feature_df))

# knn item-based cf recommender
matrix = merged_dataset.pivot_table(index='Game_Name', columns='User_ID', values='Hours').fillna(0)

rating_matrix = csr_matrix(matrix.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(rating_matrix)

#query_index = np.random.choice(matrix.shape[0])
query_name = 'Portal 2'

result = 0
for i in range(matrix.shape[0]):
    if(query_name == matrix.index[i]):
        result = i
        break

distances, indices = model_knn.kneighbors(matrix.iloc[result, :].values.reshape(1, -1), n_neighbors = 11)

print("Indices: {0}".format(indices))
print("Distances: {0}".format(distances))

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(matrix.index[result]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, matrix.index[indices.flatten()[i]], distances.flatten()[i]))


plt.show()



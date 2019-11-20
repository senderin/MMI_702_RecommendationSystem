from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

list1 = ["My name is xyz", "My name is pqr", "I work in abc"]
list2 = ["My name is xyz", "I work in abc"]

vectorizer = TfidfVectorizer(min_df = 0, max_df=0.5, stop_words = "english", ngram_range = (1,3))
vec = vectorizer.fit(list1)   # train vec using list1
vectorized = vec.transform(list1)   # transform list1 using vec

km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, n_jobs=1)
features = vectorizer.get_feature_names()
km.fit(vectorized)
list2Vec = vec.transform(list2)  # transform list2 using vec
clusters = km.predict(list2Vec)

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(2):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :2]:
        print(' %s' % features[ind], end='')
    print()
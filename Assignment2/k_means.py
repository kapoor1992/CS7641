from utilities.data_fetcher import *
from sklearn.cluster import KMeans

train_count, train_attributes, train_labels, test_attributes, test_labels = get_mammography_data(100)

kmeans = KMeans(n_clusters=2, random_state=0).fit(train_attributes)
print(kmeans.cluster_centers_)
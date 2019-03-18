from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from utilities.data_fetcher import *
import pca, ica, random_projection, factor_an
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def run(X, clust_ct=None):
    if clust_ct is None:
        range_n_clusters = [2, 3, 4, 5, 6, 7]
    else:
        range_n_clusters = [clust_ct]

    for n_clusters in range_n_clusters:
        fig, ax1 = plt.subplots(1, 1)

        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = GaussianMixture(n_components=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        if clust_ct is not None:
            return cluster_labels

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("n_clusters=", n_clusters,
            "silhouette_score=", silhouette_avg)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

            y_lower = y_upper + 10

        ax1.set_title("Silhouette Plot")
        ax1.set_xlabel("Coefficient Values")
        ax1.set_ylabel("Cluster Label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    if clust_ct is None:
        plt.show()

    return None
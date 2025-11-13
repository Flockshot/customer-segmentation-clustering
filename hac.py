import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

distances = ['euclidean', 'cosine']
linkages = ['single', 'complete']
k_values = [2, 3, 4, 5]

stats = []

for distance in distances:
    for linkage in linkages:

        hac = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage, metric=distance)
        hac.fit(dataset)

        plot_dendrogram(hac, truncate_mode="level", p=3)
        plt.title("Dendrogram for %s linkage and %s distance" % (linkage, distance))
        plt.xlabel('Number of points in the clusters')
        plt.ylabel('Distance')
        plt.show()

        for k in k_values:
            one_start_time = time.time()
            hac = AgglomerativeClustering(n_clusters=k, metric=distance, linkage=linkage)
            predicted = hac.fit_predict(dataset)

            silhouette_avg = silhouette_score(dataset, predicted)
            sample_silhouette_values = silhouette_samples(dataset, predicted)
            stats.append([distance, linkage, k, silhouette_avg])

            # Silhouette plot
            # The silhouette coefficient can range from -1, 1

            #plt.axis.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            #plt.axis.set_ylim([0, len(dataset) + (k + 1) * 10])

            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[predicted == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k)
                plt.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            plt.title("Silhouette plot for Distance: %s - Linkage: %s - K: %d" % (distance, linkage, k))
            plt.xlabel("The silhouette coefficient values")
            plt.ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            plt.axvline(x=silhouette_avg, color="red", linestyle="--")

            plt.yticks([])  # Clear the yaxis labels / ticks
            plt.xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.show()
            print("Statistics for Distance: %s - Linkage: %s - K: %d" % (distance, linkage, k))
            print("Average Silhouette Score: %.2f" % silhouette_avg)

            print("Time Taken: %.2f seconds" % (time.time() - one_start_time))
            print("------------------------------------------------------------")

stats.sort(key=lambda x: x[3], reverse=True)
print("Statistics for best result: ")
print("Distance: %s - Linkage: %s - K: %d" % (stats[0][0], stats[0][1], stats[0][2]))
print("Average Silhouette Score: %.2f" % stats[0][3])

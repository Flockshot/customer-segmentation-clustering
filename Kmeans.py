from Distance import Distance
import numpy as np
import random


class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.previous_centers = {i: None for i in range(K)}

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        loss = 0
        for cluster_data in self.clusters.items():
            for data in cluster_data[1]:
                loss += np.sum(
                    np.square(np.linalg.norm(np.array(data) - np.array(self.cluster_centers[cluster_data[0]]))))
        return loss

    def run(self):
        """Kmeans algorithm implementation"""

        for i in range(self.K):
            rand_index = random.randint(0, self.dataset.shape[0] - 1)
            self.cluster_centers[i] = self.dataset[rand_index]

        while True:

            try:
                np.testing.assert_equal(self.previous_centers, self.cluster_centers)
                break
            except AssertionError:
                self.previous_centers = self.cluster_centers.copy()
                self.clusters = {i: [] for i in range(self.K)}
            # if np.all(self.previous_centers.values() == self.cluster_centers.values()):
            #    break

            for data in self.dataset:
                min_dist = float('inf')
                min_index = -1
                for cluster_center_index in self.cluster_centers.keys():
                    dist = np.linalg.norm(data - self.cluster_centers[cluster_center_index])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = cluster_center_index
                self.clusters[min_index].append(data)

            for cluster_index in self.clusters.keys():
                if self.clusters[cluster_index]:
                    self.cluster_centers[cluster_index] = np.mean(np.array(self.clusters[cluster_index]), axis=0)

        return self.cluster_centers, self.clusters, self.calculateLoss()

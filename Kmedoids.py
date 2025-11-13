import random
import Distance
import numpy as np


class KMemoids:
    def __init__(self, dataset, K=2, distance_metric="cosine"):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        self.distance_metric = distance_metric
        # each cluster is represented with an integer index self.clusters stores the data points of each cluster in a
        # dictionary In this dictionary, you can keep either the data instance themselves or their corresponding
        # indices in the dataset (self.dataset).
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_medoids stores the cluster medoid for each cluster in a dictionary # In this dictionary,
        # you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.cluster_medoids = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.medoids_list = []
        self.previous_cost = 0
        self.cost = 0

    def calculateLoss(self):
        """Loss function implementation of Equation 2"""
        loss = 0
        for cluster_data in self.clusters.items():
            for data in cluster_data[1]:
                loss += Distance.Distance.calculateCosineDistance(data, self.cluster_medoids[cluster_data[0]])
        return loss

    def run(self):
        """Kmedoids algorithm implementation"""
        for i in range(self.K):
            rand_index = random.randint(0, self.dataset.shape[0] - 1)
            self.cluster_medoids[i] = self.dataset[rand_index]
            self.medoids_list.append(self.dataset[rand_index])
        i = 1
        self.setClusters()

        while self.cost <= self.previous_cost:



            self.cost = 0
            self.previous_cost = self.calculateCost()

            to_break = False
            for cluster_index in self.clusters.keys():
                medoid = self.cluster_medoids[cluster_index]
                for data in self.clusters[cluster_index]:
                    if np.any(data != medoid):

                        #print("Medoid: ", medoid)
                        #print("Data: ", data)
                        #print("List: ", self.medoids_list)

                        to_continue = False
                        for prev_meds in self.medoids_list:
                            if np.all(prev_meds == data):
                                to_continue = True
                                break
                        if to_continue:
                            continue

                        self.previous_cost = self.calculateCost()

                        self.cluster_medoids[cluster_index] = data
                        self.medoids_list.append(data)
                        old_cluster = self.clusters
                        self.clusters = {i: [] for i in range(self.K)}
                        self.setClusters()

                        self.cost = self.calculateCost()

                        #print("Iteration: ", i)
                        #print("Cost: ", self.cost)
                        #print("Previous Cost: ", self.previous_cost)

                        if self.cost > self.previous_cost:
                            self.cluster_medoids[cluster_index] = medoid
                            self.clusters = old_cluster
                        to_break = True
                        break

                if to_break:
                    break
            # print("Iteration: ", i)
            i += 1

        return self.cluster_medoids, self.clusters, self.calculateLoss()

    def setClusters(self):
        for data in self.dataset:
            min_dist = float('inf')
            min_index = -1
            for cluster_center_index in self.cluster_medoids.keys():
                dist = Distance.Distance.calculateCosineDistance(data, self.cluster_medoids[
                    cluster_center_index])
                if dist < min_dist:
                    min_dist = dist
                    min_index = cluster_center_index
            self.clusters[min_index].append(data)

    def calculateCost(self):
        total_cost = 0
        for medoid_index in self.cluster_medoids.keys():
            medoid_cost = 0
            for cost_data in self.clusters[medoid_index]:
                if np.any(cost_data != self.cluster_medoids[medoid_index]):
                    medoid_cost += Distance.Distance.calculateCosineDistance(cost_data,
                                                                             self.cluster_medoids[medoid_index])
            total_cost += medoid_cost
        return total_cost

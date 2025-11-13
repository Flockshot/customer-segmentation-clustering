from Kmedoids import KMemoids
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt



def calc_lower_bound(mean, std):
    return mean - ((1.96 * std) / np.sqrt(10))


def calc_upper_bound(mean, std):
    return mean + ((1.96 * std) / np.sqrt(10))


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

datasets = [dataset1, dataset2]
k_values = [k for k in range(2, 11)]

start_time = time.time()

stats = []
dataset_num = 0

for dataset in datasets:
    dataset_num += 1
    k_losses = []
    for k in k_values:
        lowest_losses = []
        one_start_time = time.time()
        for i in range(10):
            losses = []
            for j in range(10):
                k_means = KMemoids(dataset, k)
                centers, clusters, loss = k_means.run()
                losses.append(loss)
            lowest_losses.append(min(losses))

        mean = np.mean(lowest_losses)
        std = np.std(lowest_losses)
        lower_bound = calc_lower_bound(mean, std)
        upper_bound = calc_upper_bound(mean, std)

        stats.append([dataset_num, k, mean, std, lower_bound, upper_bound])

        k_losses.append(mean)

        print("Statistics for Dataset: %d - K: %d" % (dataset_num, k))
        print("Mean: %.2f" % mean)
        print("Std: %.2f" % std)
        print("Lower Bound: %.2f" % lower_bound)
        print("Upper Bound: %.2f" % upper_bound)
        print("Time Taken: %.2f seconds" % (time.time() - one_start_time))
        print("------------------------------------------------------------")

    plt.plot(k_values, k_losses)
    plt.title('Loss vs K for Dataset %d' % dataset_num)
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.show()

elapsed_time = time.time() - start_time
print("Total Time Taken: %.2f seconds" % elapsed_time)
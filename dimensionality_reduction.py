import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

datasets = [dataset1, dataset2]
distances = ['euclidean', 'cosine']
methods = {"TSNE": TSNE, "UMAP": UMAP}

dataset_num = 0
for dataset in datasets:
    dataset_num += 1
    for distance in distances:
        for name, method in methods.items():
            reduction = method(metric=distance)
            values = reduction.fit_transform(dataset)
            plt.scatter(values[:, 0], values[:, 1])
            plt.title("Scatter plot for %s reduction and %s distance for Dataset %d" % (name, distance, dataset_num))
            plt.show()

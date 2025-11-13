# Unsupervised Clustering for Customer Segmentation

Implemented, evaluated, and compared multiple unsupervised clustering algorithms to uncover hidden patterns and customer groupings in unlabeled datasets. The goal was to identify optimal clustering structures and validate them through both quantitative metrics and visualization.

![Python](https://img.shields.io/badge/Python-NumPy_&_SciPy-blue.svg?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-informational.svg?logo=matplotlib&logoColor=white)

## Methodology

This project implemented and compared three primary families of clustering algorithms on multiple datasets.

1.  **K-Means Clustering (from Scratch):**
    * Implemented the K-Means algorithm from scratch (`Kmeans.py`) using NumPy.
    * [cite_start]The algorithm iteratively assigns points to the nearest cluster (E-step) and recalculates the cluster's centroid (M-step) until convergence [cite: 1354-1355, 1483-1492].

2.  **K-Medoids Clustering (from Scratch):**
    * [cite_start]Implemented the K-Medoids algorithm from scratch (`Kmedoids.py`), which is more robust to outliers than K-Means [cite: 1365-1366, 1403, 1493-1545].
    * [cite_start]This implementation was specifically designed to use **Cosine Distance**, making it effective for high-dimensional or non-Euclidean data[cite: 1537].

3.  **Hierarchical Agglomerative Clustering (HAC):**
    * [cite_start]Implemented HAC using `sklearn.cluster.AgglomerativeClustering` to provide a hierarchical (bottom-up) alternative [cite: 1583-1584, 1622].
    * Systematically tested four combinations of linkage criteria and distance metrics:
        * Single Linkage + Euclidean Distance
        * Single Linkage + Cosine Distance
        * Complete Linkage + Euclidean Distance
        * Complete Linkage + Cosine Distance

## Evaluation & Validation

A multi-step process was used to determine the optimal number of clusters (K) and validate the quality of the final groupings.

1.  **Elbow Method (Finding K):**
    * To find the best `K` for K-Means and K-Medoids, a rigorous experiment was run (`kmeans_experiment.py`, `kmedoids_experiment.py`).
    * For each K from 2 to 10, the algorithm was run 10 times, and the lowest loss was recorded.
    * [cite_start]This process was *itself repeated 10 times*, and the results were averaged to find a statistically stable loss value for each K [cite: 5, 101, 1548-1550].
    * The resulting "Loss vs. K" graph was used to identify the "elbow point," where the rate of loss reduction sharply decreases.

    > **[Image: Elbow Plot for K-Means on Dataset 1]**
    >
    > *(**Developer Note:** Place your elbow plot from `report.pdf` (Page 1) here. It clearly shows the elbow at K=5.)*

2.  **Silhouette Analysis (Validating K):**
    * [cite_start]For HAC, Silhouette Analysis was used to find the optimal K (from 2, 3, 4, 5) for each of the four linkage/distance combinations [cite: 1597-1619, 1626-1628].
    * This method measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). A score close to 1 indicates good clustering.

    > **[Image: Silhouette Plot for the Best HAC Configuration]**
    >
    > *(**Developer Note:** Place your best silhouette plot from `report.pdf` (Page 15) here. This plot provides quantitative proof of your best result.)*

3.  **Dimensionality Reduction (Visual Confirmation):**
    * To visually inspect the clusters in 2D, **t-SNE** and **UMAP** were applied to the datasets (`dimensionality_reduction.py`).
    * These visualizations confirmed that the K values identified by the Elbow and Silhouette methods corresponded to visually distinct groupings of data points.

    > **[Image: t-SNE Scatter Plot of Dataset 1]**
    >
    > *(**Developer Note:** Place your t-SNE plot from `report.pdf` (Page 5) here. It visually confirms the 5 clusters.)*

## 📊 Key Results

The analysis successfully identified optimal cluster counts for the provided datasets:

* **Dataset 1 (K-Means & K-Medoids):** The Elbow method identified **K = 5** as the optimal number of clusters. This was visually confirmed by t-SNE and UMAP, which both showed five distinct groupings.
* **Dataset 2 (K-Means & K-Medoids):** The Elbow method identified **K = 4** as the optimal number of clusters. This was also confirmed by the 2D visualizations.
* **HAC Analysis:** The best clustering configuration found by Silhouette Analysis was **Euclidean distance, single linkage, and K = 4**, which achieved an **Average Silhouette Score of 0.77**.

## 🚀 How to Run

### Requirements
* Python 3
* `numpy`
* `matplotlib`
* `scikit-learn`
* `umap-learn`
* `pickle` (for loading data)

### Running the Experiments

The project is divided into scripts. To run the analyses, execute the experiment files from the root directory.

```bash
# To run the K-Means elbow method experiment
python kmeans_experiment.py

# To run the K-Medoids (cosine distance) elbow method experiment
python kmedoids_experiment.py

# To run the Hierarchical Clustering (HAC) + Silhouette analysis
python hac.py

# To generate the 2D visualizations for Part 2 datasets
python dimensionality_reduction.py
```
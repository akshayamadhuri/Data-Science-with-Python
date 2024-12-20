# Unsupervised Learning: Hands-on with K-means Clustering

## Quick Recap: Unraveling K-means Clustering

The main principle of K-means clustering is quite simple: it groups data points into distinct clusters based on their mutual distances to minimize the variance, also known as inertia, within each cluster.

We will now apply K-means clustering to a well-known dataset: the Iris dataset.

## Diving into the Iris Dataset Again

The Iris dataset, as we've discussed in previous lessons, consists of measurements taken from 150 iris flowers across three distinct species. Imagine being a botanist searching for a systematic way to categorize new iris flowers based on these features. Doing so manually would be burdensome; hence, resorting to machine learning, specifically K-means clustering, becomes a logical choice!

Let's load this dataset using the sklearn library in Python and convert it into a pandas DataFrame:

```Python

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df.head()
```

## Implementing K-means Clustering with sklearn

We're now going to implement K-means clustering on the Iris dataset. For this, we'll use the KMeans class from sklearn's cluster module. To keep our initial implementation straightforward, let's focus on just two dataset features: sepal length and sepal width.

```Python

from sklearn.cluster import KMeans

# Assigning the features for our model
X = iris_df.iloc[:, [0, 1]].values

# Defining the KMeans clustering model
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
```
In the above block:

`n_clusters`: stipulates the number of clusters to form.
`init`: sets the method for initialization. The "k-means++" method selects initial cluster centers intelligently to hasten convergence.
`max_iter`: limits the maximum number of iterations for a single run.
`n_init`: specifies the number of times the algorithm runs with different centroid seeds.

We have left out parameters like tol and algorithm:

tol is the tolerance with regard to inertia required to declare convergence. We did not include this to use the default tolerance.
algorithm specifies the algorithm to use. The classical EM-style algorithm is "full", the Elkan variant is more efficient but is not available for sparse data. To keep things simple, we chose not to specify this option.

## Visualizing the Clusters

Next, let's visualize our data points and their respective clusters using matplotlib, a powerful plotting library in Python. This visualization will help us better understand our K-means clustering and evaluate how well it performs:

```Python

import matplotlib.pyplot as plt

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')

plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/a67eff8a-64f2-422b-acd8-3234ad904d6c)



This code plots the data points, coloring each one according to the cluster it belongs to. The centroids, the centers of each cluster, are also plotted as yellow stars. A visual inspection shows that our clustering successfully differentiated between the different species of Iris flowers.

## Assessing Cluster Quality with The Silhouette Score

An integral part of implementing K-means clustering is evaluating the effectiveness of the formed clusters. The silhouette_score provides a quantifiable way to accomplish this. It measures how close each data point in one cluster is to the data points in the neighboring clusters. This score ranges from -1 to +1. A high value indicates that the data point fits well within its own cluster and is poorly matched to neighboring clusters.

Let's calculate this score using the following code:

```Python

from sklearn.metrics import silhouette_score

score = silhouette_score(X, y_kmeans)
print(f'Silhouette Score(n=3): {silhouette_score(X, y_kmeans)}')
# Silhouette Score(n=3): 0.4450525692083638
```
The silhouette_score can guide us to understand how well our data points have been clustered.


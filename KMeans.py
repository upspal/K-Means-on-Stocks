import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from niftystocks import ns
from sklearn.decomposition import PCA
from IPython.display import clear_output


def random_centroids(scaled_data,k):
    centroids = []
    for i in range(k):
        centroid = scaled_data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids,axis=1)    

def get_labels(scaled_data, centroids):
    distances = centroids.apply(lambda x : np.sqrt(((scaled_data - x)**2).sum(axis=1)))
    return distances.idxmin(axis=1)

def new_centroids(scaled_data, labels, k):
    centroids = scaled_data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

def plot_clusters(scaled_data, labels, centroids, interation):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(scaled_data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title("Iteration: {}".format(interation))
    plt.scatter(data_2d[:,0], data_2d[:,1], c=labels)
    plt.scatter(centroids_2d[:,0], centroids_2d[:,1], c='r', s=100)
    plt.show()

get_nifty50 = ns.get_nifty50_with_ns()
data = yf.download(get_nifty50, start="2020-01-01", end="2021-01-01")

returns = data['Adj Close'].pct_change()
returns = returns.iloc[1:]
returns = returns.dropna(axis=1)

features = ["Mean Returns", "Volatility", "Sharpe Ratio"]
cluster_data = pd.DataFrame(index=returns.columns, columns=features)
cluster_data["Mean Returns"] = returns.mean()
cluster_data["Volatility"] = returns.std()
cluster_data["Sharpe Ratio"] = cluster_data["Mean Returns"] / cluster_data["Volatility"]

cluster_data = cluster_data.dropna(subset=features)

scaled_data = ((cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min())) * 9 + 1

max_iterations = 100
k = 4

centroids = random_centroids(scaled_data, k)
old_centroids = pd.DataFrame() 
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_labels(scaled_data, centroids)
    centroids = new_centroids(scaled_data,labels,k)
    plot_clusters(scaled_data, labels, centroids, iteration)
    iteration += 1
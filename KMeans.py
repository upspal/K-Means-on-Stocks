import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from niftystocks import ns
import matplotlib.pyplot as plt

def random_centroids(scaled_data, k):
    centroids = []
    for i in range(k):
        centroid = scaled_data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

def get_labels(scaled_data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((scaled_data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

def new_centroids(scaled_data, labels, k):
    centroids = scaled_data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

def plot_clusters(scaled_data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(scaled_data)
    centroids_2d = pca.transform(centroids.T)
    plt.title("Iteration: {}".format(iteration))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', s=50, label='Centroids')
    unique_labels = sorted(set(labels))
    legend_elements = []
    for idx, label in enumerate(unique_labels):
        mask = (labels == label)
        if any(mask):
            color = scatter.to_rgba(label)  # Get exact color used for this label
            legend_elements.append(plt.scatter([], [], c=[color], label=f'Cluster {idx+1}'))
    # Add centroids to legend and display
    plt.legend(handles=legend_elements + [plt.scatter([], [], c='red', s=50, label='Centroids')])
    st.pyplot(plt)

def KMeansClustering(scaled_data, k):
    centroids = random_centroids(scaled_data, k)
    old_centroids = pd.DataFrame() 
    iteration = 1
    max_iterations = 100

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(scaled_data, centroids)
        centroids = new_centroids(scaled_data,labels,k)
        iteration += 1
    plot_clusters(scaled_data, labels, centroids, iteration)
    # Map cluster labels to 1-based sequential numbers
    unique_labels = sorted(set(labels))
    label_mapping = {old: new+1 for new, old in enumerate(unique_labels)}
    mapped_labels = labels.map(label_mapping)
    scaled_data = pd.concat([scaled_data, pd.Series(mapped_labels, name='Cluster')], axis=1)
    return scaled_data        

def main():
    st.set_page_config(
    page_title="K-Means Clustering for Stocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

    st.title("K-Means Clustering for Stocks")
    st.sidebar.title("📈 K-Means Clustering for Stocks")
    st.sidebar.write("Created by")
    linkedin = "https://www.linkedin.com/in/pranavuppall"
    st.markdown(f'<a href="{linkedin}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Pranav Uppal`</a>', unsafe_allow_html=True)

    # Clustering Parameters
    st.sidebar.header("Clustering Parameters")
    k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=4)
    timeframe_start = st.sidebar.date_input("Timeframe(Start)", pd.to_datetime("2020-01-01"), max_value=pd.to_datetime("2023-01-01"))
    timeframe_end = st.sidebar.date_input("Timeframe(End)", pd.to_datetime("2021-01-01"), max_value=pd.to_datetime("2023-01-01"))

    # Stock Selection
    st.sidebar.header("Stock Selection")
    custom_ticker = st.sidebar.text_input("Enter Custom Ticker (separated by space):")
    select_all = st.sidebar.checkbox("Select All Stocks",value=True)
    selected_stocks = []
    if select_all:
        selected_stocks = ns.get_nifty50_with_ns()
    else:
        selected_stocks = st.sidebar.multiselect("Select Stock", ns.get_nifty50_with_ns())
    if custom_ticker:
        custom_stocks = [stock.strip() for stock in custom_ticker.split(' ')]
        selected_stocks.extend(custom_stocks)

    if not selected_stocks:
        st.warning("Please select at least one stock to proceed.")
        return
    
    st.sidebar.header("Rerun Clustering")
    st.sidebar.write("The clustering can generate different results per run as each iteration reaches a local minimum!")
    if st.sidebar.button("Rerun"):
        st.cache_data.clear()

    # Data Fetching and Preprocessing
    try:
        if timeframe_start == timeframe_end:
            st.error("Start and end dates are the same. Please select different dates.")
            return
        
        data = yf.download(selected_stocks, start=timeframe_start, end=timeframe_end)
        
        if data.empty:
            st.error("No data found for the selected stocks. Please check if the stock symbols are correct.")
            return
            
        returns = data['Adj Close'].pct_change()
        returns = returns.iloc[1:]
        returns = returns.dropna(axis=1)
        
        if returns.empty:
            st.error("No valid return data available for the selected stocks and timeframe.")
            return

        if len(returns.columns) < k:
            st.error(f"Number of valid stocks ({len(returns.columns)}) is less than the number of clusters ({k}). Please select more stocks or reduce the number of clusters.")
            return

        features = ["Mean Returns", "Volatility", "Sharpe Ratio"]
        cluster_data = pd.DataFrame(index=returns.columns, columns=features)
        cluster_data["Mean Returns"] = returns.mean()
        cluster_data["Volatility"] = returns.std()
        cluster_data["Sharpe Ratio"] = cluster_data["Mean Returns"] / cluster_data["Volatility"]

        cluster_data = cluster_data.dropna(subset=features)

        scaled_data = ((cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min())) * 9 + 1

        col1, col2 = st.columns(2)
        with col1:
            st.header("Clustered Stock Returns")
            rawdataset = KMeansClustering(scaled_data, k)
            dataset=rawdataset.dropna(subset=['Cluster'])

        with col2:
            st.header("Clustered Stock Returns Data")
            dataset=cluster_data.join(dataset['Cluster']) 
            st.write(dataset)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if "Invalid ticker" in str(e):
            st.error("One or more stock symbols are invalid. Please check your input.")

if __name__ == "__main__":
    main()

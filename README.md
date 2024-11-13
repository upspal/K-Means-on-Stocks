# K-Means Clustering on Stocks

This project involves the manual implementation of the K-Means clustering algorithm applied to stocks. The primary objective of this project was to conduct personal research and gain a deeper understanding of clustering techniques and their applications in financial data analysis.

[[https://kmeans-on-stocks.streamlit.app/]]

## Project Overview

The K-Means algorithm is a popular unsupervised learning method used for clustering data into distinct groups based on their features. In this project, I manually implemented the K-Means algorithm from scratch, without relying on pre-built libraries, to better understand its inner workings and nuances.

![Kmeans-in-action](https://i.imgur.com/iDtx9Wh.gif)

## Features
1. **Interactive Stock Selection**:
   - Users can select multiple stocks from the NIFTY 50 index and add custom stocks for analysis
   - Flexible timeframe selection to analyze different time periods
   - Ability to specify the number of clusters (k) for the analysis

2. **Real-time Clustering**:
   - Dynamic clustering of selected stocks based on user parameters
   - Visual representation of clusters using dimensionality reduction (PCA)
   - Interactive visualization showing cluster assignments and centroids

3. **Data Analysis Tools**:
   - Detailed view of financial metrics for each stock
   - Cluster-wise analysis showing common characteristics
   - Visual tracking of clustering iterations and convergence

4. **User-friendly Interface**:
   - Clean and intuitive Streamlit-based web interface
   - Real-time updates as parameters are modified
   - Easy-to-interpret visualizations of clustering results

## Implementation Details

1. **Data Collection**: 
   - Historical stock price data was collected for NIFTY 50 stocks using the `yfinance` library.
   - The data includes daily closing prices, which were used to calculate various financial metrics.

2. **Data Preprocessing**:
   - The collected data was cleaned and preprocessed to handle missing values and normalize the features.
   - From the data, we derived the following features: Mean Returns, Volatility, and Sharpe Ratio.

3. **Manual K-Means Implementation**:
   - Initialize the centroids randomly from the data points.
   - The algorithm iteratively assigned each data point to the nearest centroid and updated the centroids based on the mean of the assigned points.
   - The process was repeated until the centroids stabilized or a maximum number of iterations was reached.

## How to Run

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `KMeans.ipynb` to see the implementation and results.
4. Launch the Streamlit application using the command `streamlit run KMeans.py` to interact with the clustering results.



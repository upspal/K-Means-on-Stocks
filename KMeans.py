import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

nifty50_tickers = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'TATAMOTORS.NS', 'ULTRACEMCO.NS',
    'BAJAJFINSV.NS', 'WIPRO.NS', 'NTPC.NS', 'POWERGRID.NS', 'ADANIPORTS.NS',
    'JSWSTEEL.NS', 'TECHM.NS', 'ADANIENT.NS', 'TATASTEEL.NS', 'NESTLEIND.NS',
    'M&M.NS', 'ONGC.NS', 'GRASIM.NS', 'INDUSINDBK.NS', 'HINDALCO.NS',
    'CIPLA.NS', 'DRREDDY.NS', 'BAJAJ-AUTO.NS', 'COALINDIA.NS', 'EICHERMOT.NS',
    'SBILIFE.NS', 'HDFCLIFE.NS', 'BRITANNIA.NS', 'TATACONSUM.NS', 'APOLLOHOSP.NS',
    'HEROMOTOCO.NS', 'DIVISLAB.NS', 'UPL.NS', 'BPCL.NS', 'SHREECEM.NS'
]

def plot_clusters(scaled_data, labels, centroids, window_name):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(scaled_data)
    centroids_2d = pca.transform(centroids)

    fig = plt.figure(figsize=(10, 8))
    plt.title(f"Stock Clusters for {window_name}")

    # Plot each cluster with a different color
    unique_labels = np.unique(labels)
    for i in unique_labels:
        plt.scatter(
            data_2d[labels == i, 0],
            data_2d[labels == i, 1],
            label=f'Cluster {i+1}'
        )

    # Plot centroids
    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c='red',
        marker='X',
        s=100,
        label='Centroids'
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    return fig

def run_kmeans_for_window(data, window, k=4):
    start_date = window['start']
    end_date = window['end']
    window_name = window['name']

    window_data = data.loc[start_date:end_date]

    returns = window_data['Close'].pct_change().dropna()

    returns = returns.dropna(axis=1)

    features = ["Mean Returns", "Volatility", "Sharpe Ratio", "Max Drawdown", "Rolling_50_200_Ratio"]
    cluster_data = pd.DataFrame(index=returns.columns)

    cluster_data["Mean Returns"] = returns.mean() * 252  # Annualized
    cluster_data["Volatility"] = returns.std() * np.sqrt(252)  # Annualized
    cluster_data["Sharpe Ratio"] = cluster_data["Mean Returns"] / cluster_data["Volatility"]

    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / rolling_max) - 1
    cluster_data["Max Drawdown"] = drawdowns.min()

    # Calculate 50-day and 200-day moving averages ratio
    prices = window_data['Close']
    for ticker in prices.columns:
        if ticker in cluster_data.index:
            ma_50 = prices[ticker].rolling(window=50).mean()
            ma_200 = prices[ticker].rolling(window=200).mean()
            # Use the last value of the ratio
            if not ma_200.empty and not ma_50.empty and ma_200.iloc[-1] and ma_50.iloc[-1] and not np.isnan(ma_200.iloc[-1]) and not np.isnan(ma_50.iloc[-1]):
                cluster_data.loc[ticker, "Rolling_50_200_Ratio"] = ma_50.iloc[-1] / ma_200.iloc[-1]

    cluster_data = cluster_data.dropna()

    scaled_data = ((cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min())) * 9 + 1

    # Run K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    result_data = cluster_data.copy()
    result_data['Cluster'] = labels + 1  # Make clusters 1-based

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=scaled_data.columns)

    return {
        'window': window_name,
        'results': result_data,
        'centroids': kmeans.cluster_centers_,
        'features': features,
        'plot_fig': plot_clusters(scaled_data, labels, centroids, window_name)
    }

def create_report(all_results, k=4):
    report = pd.DataFrame()

    for result in all_results:
        window_name = result['window']
        window_results = result['results']

        for cluster_id in range(1, k+1):
            cluster_stocks = window_results[window_results['Cluster'] == cluster_id]

            if cluster_stocks.empty:
                continue

            for stock in cluster_stocks.index:
                stock_data = {
                    'Year': window_name,
                    'Stock': stock,
                    'Cluster': f"Cluster_{cluster_id}",
                    'Mean_Returns': cluster_stocks.loc[stock, 'Mean Returns'],
                    'Volatility': cluster_stocks.loc[stock, 'Volatility'],
                    'Sharpe_Ratio': cluster_stocks.loc[stock, 'Sharpe Ratio'],
                    'Max_Drawdown': cluster_stocks.loc[stock, 'Max Drawdown'],
                    'MA_50_200_Ratio': cluster_stocks.loc[stock, 'Rolling_50_200_Ratio']
                }
                report = pd.concat([report, pd.DataFrame([stock_data])], ignore_index=True)

    cluster_stats = report.groupby(['Year', 'Cluster']).agg({
        'Mean_Returns': ['mean', 'min', 'max'],
        'Volatility': ['mean', 'min', 'max'],
        'Sharpe_Ratio': ['mean', 'min', 'max'],
        'Max_Drawdown': ['mean', 'min', 'max'],
        'MA_50_200_Ratio': ['mean', 'min', 'max'],
        'Stock': 'count'
    }).reset_index()

    cluster_stats.columns = [
        'Year', 'Cluster',
        'Mean_Returns_Avg', 'Mean_Returns_Min', 'Mean_Returns_Max',
        'Volatility_Avg', 'Volatility_Min', 'Volatility_Max',
        'Sharpe_Ratio_Avg', 'Sharpe_Ratio_Min', 'Sharpe_Ratio_Max',
        'Max_Drawdown_Avg', 'Max_Drawdown_Min', 'Max_Drawdown_Max',
        'MA_Ratio_Avg', 'MA_Ratio_Min', 'MA_Ratio_Max',
        'Stock_Count'
    ]

    return report, cluster_stats

def create_transitions(stock_report):
    stock_transitions = {}
    for stock in stock_report['Stock'].unique():
        stock_data = stock_report[stock_report['Stock'] == stock]
        if len(stock_data) > 1:  # Only include stocks with data for multiple years
            transitions = []
            for year in range(2020, 2025):
                year_data = stock_data[stock_data['Year'] == str(year)]
                if not year_data.empty:
                    cluster = year_data['Cluster'].values[0]
                    transitions.append(cluster)
                else:
                    transitions.append('NA')
            stock_transitions[stock] = transitions

    transitions_df = pd.DataFrame.from_dict(stock_transitions, orient='index',
                                           columns=['2020', '2021', '2022', '2023', '2024'])
    transitions_df.index.name = 'Stock'
    transitions_df.reset_index(inplace=True)
    return transitions_df

def main():
    st.set_page_config(
        page_title="K-Means Clustering for Nifty50 Stocks",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("K-Means Clustering for Nifty50 Stocks (Yearly Analysis)")
    st.sidebar.title("📈 Clustering Parameters")
    st.sidebar.write("Created by")
    linkedin = "https://www.linkedin.com/in/pranavuppall"
    st.sidebar.markdown(f'<a href="{linkedin}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Pranav Uppal`</a>', unsafe_allow_html=True)

    # Parameters
    k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=4)
    download_start = st.sidebar.date_input("Download Start Date", pd.to_datetime("2019-01-01"))
    download_end = st.sidebar.date_input("Download End Date", pd.to_datetime("2025-01-01"))

    # Stock Selection
    st.sidebar.header("Stock Selection")
    select_all = st.sidebar.checkbox("Select All Nifty50 Stocks", value=True)
    selected_stocks = nifty50_tickers if select_all else st.sidebar.multiselect("Select Stocks", nifty50_tickers)

    if not selected_stocks:
        st.warning("Please select at least one stock to proceed.")
        return

    # Run button
    run_analysis = st.sidebar.button("Run Analysis")

    if run_analysis:
        with st.spinner("Downloading Nifty50 data..."):
            data = yf.download(selected_stocks, start=download_start, end=download_end, progress=False)
        st.success("Download complete!")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title
st.title("Customer Segmentation using K-Means")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose columns", df.columns)

    if len(features) >= 2:
        X = df[features]

        # Choose number of clusters
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

        # Elbow Method
        st.subheader("Elbow Method")

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title("Elbow Method")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        y_pred = kmeans.fit_predict(X)

        df["Cluster"] = y_pred

        st.subheader("Clustered Data")
        st.write(df.head())

        # Visualization (only for 2 features)
        if len(features) == 2:
            fig2, ax2 = plt.subplots()
            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='viridis')
            ax2.scatter(kmeans.cluster_centers_[:, 0],
                        kmeans.cluster_centers_[:, 1],
                        s=300, c='red', label='Centroids')
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            ax2.legend()
            st.pyplot(fig2)

    else:
        st.warning("Please select at least 2 features")
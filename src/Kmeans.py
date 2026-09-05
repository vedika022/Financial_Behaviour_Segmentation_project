
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import joblib
# from sklearn.preprocessing import MinMaxScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = ROOT_DIR / "Data" / "customers_processed.csv"
df = pd.read_csv(PROCESSED_DATA_PATH)

features = df[['Annual Income ($)', 'Spending Score (1-100)']]

X = features.to_numpy()

# Using elbow method

inertia = []

for k in range(2, 11):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10 )
    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

# Using Silhoutte score for different k values

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans( n_clusters=k, random_state=42, n_init=10 )
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels )
    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
        )

# Plotting Silhoutte score and Elbow method to find optimal k

plt.figure(figsize=(20, 6))

plt.subplot(1,2,1)
plt.plot(range(2, 11), inertia, marker="o", linewidth=2 )

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

plt.xticks(range(2, 11))
plt.grid(True)

plt.subplot(1,2,2)
plt.plot( range(2, 11), silhouette_scores, marker="o", linewidth=2 )

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")

plt.grid(True)

plt.show()

# By analysing both graphs the optimal k value is 8

optimal_k = 8 

# Final k means model 

kmeans = KMeans( n_clusters=optimal_k, random_state=42, n_init=10 )

df["Cluster"] = kmeans.fit_predict(X)

# Cluster counts

cluster_counts = ( df["Cluster"].value_counts().sort_index() )
print(cluster_counts)

# Visualize the Customer Clusters

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Annual Income ($)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=80,
    alpha=0.8
)

plt.title(
    "Customer Segmentation: Income vs Spending Score"
)

plt.xlabel("Annual Income ($)")
plt.ylabel("Spending Score (1-100)")

plt.legend(title="Cluster")

plt.show()


#  Find the Average Income and Spending Score of Each Cluster

cluster_profile = (df.groupby("Cluster")[["Annual Income ($)", "Spending Score (1-100)"]].mean())

print(cluster_profile)

# Give Meaningful Names to the Clusters

# Cluster names
cluster_names = {
    0: "Budget Moderates",
    1: "Mid-Market Moderates",
    2: "High-Income Savers",
    3: "Affluent Moderates",
    4: "Active Spenders",
    5: "Budget-Conscious",
    6: "Premium Customers",
    7: "High-Engagement Budget"
}

# Create a new column with the segment name
df["Segment"] = df["Cluster"].map(cluster_names)

# View cluster and segment names
print(df[["Cluster", "Segment"]].drop_duplicates().sort_values("Cluster"))


# Plot K-Means Centroids

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Separate the features and convert them back to original scale
features_original = scaler.inverse_transform( df[["Annual Income ($)", "Spending Score (1-100)"]])

# Put original values back into a DataFrame
df_plot = df.copy()
df_plot[["Annual Income ($)", "Spending Score (1-100)"]] = features_original

# Convert K-Means centroids back to original scale
centroids = scaler.inverse_transform( kmeans.cluster_centers_)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df_plot,
    x="Annual Income ($)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=80,
    alpha=0.7
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=300,
    c="black",
    label="Centroids"
)

plt.title("Customer Segmentation with K-Means Centroids")
plt.xlabel("Annual Income ($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Save the Results

output_columns = [

    "Annual Income ($)",
    "Spending Score (1-100)",
    "Cluster",
    "Segment"
]

final_df = df[output_columns]

final_df.to_csv( "Data/customer_segmentation_results.csv", index=False )

# Save the trained model

joblib.dump(kmeans, "models/kmeans_model.pkl")

print("K-Means model saved successfully.")

print( "Customer segmentation completed successfully!")

print( "Results saved to customer_segmentation_results.csv")




















from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

KMEANS_DATA_PATH = ROOT_DIR / "Data" / "customerS_segmented_kmeans.csv"
GMM_DATA_PATH = ROOT_DIR / "Data" / "customers_segmented_gmm.csv"

df_km = pd.read_csv(KMEANS_DATA_PATH)
df_gmm = pd.read_csv(GMM_DATA_PATH)

# taking features 
features = df_gmm[['Annual Income ($)', 'Spending Score (1-100)']]
X = features.to_numpy()

#  Comprehensive Model Evaluation

sil_score_km = silhouette_score(X, df_km['Cluster'])
db_score_km = davies_bouldin_score(X, df_km['Cluster'])
ch_score_km = calinski_harabasz_score(X, df_km['Cluster'])

sil_score_gmm = silhouette_score(X, df_gmm['Cluster'])
db_score_gmm = davies_bouldin_score(X, df_gmm['Cluster'])
ch_score_gmm = calinski_harabasz_score(X, df_gmm['Cluster'])

comparison = {
    'Metric' : ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score'],
    'Interpretation' : ['Higher is better', 'Lower is better', 'Higher is better'],
    'K-means' : [sil_score_km, db_score_km, ch_score_km],
    'Guassian Mixture' : [sil_score_gmm, db_score_gmm, ch_score_gmm]
}

df_compare = pd.DataFrame(comparison)

print(df_compare)




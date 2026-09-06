import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

def main():
    # 1. Load Processed Data
    file_path = 'Data/customers_processed.csv'
    df = pd.read_csv(file_path)
    
    # Feature columns present in dataset
    features = ['Annual Income ($)', 'Spending Score (1-100)']
    X = df[features]

    # using aic, bic scores to determine the suitable n_components
    
    bic_scores = []
    aic_scores = []

    for k in range(2, 11):

        gmm = GaussianMixture(
            n_components=k,
            random_state=42
        )

        gmm.fit(X)

        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    best_bic_k = range(2, 11)[bic_scores.index(min(bic_scores))]
    best_aic_k = range(2, 11)[aic_scores.index(min(aic_scores))]

    print("Best n_components according to BIC:", best_bic_k)
    print("Best n_components according to AIC:", best_aic_k)

    # 2. Train Gaussian Mixture Model (GMM)

    n_components = best_bic_k

    gmm = GaussianMixture(
        n_components=n_components, 
        covariance_type='full', 
        random_state=42
    )

        # Predict hard cluster labels
    df['Cluster'] = gmm.fit_predict(X)
    

    # 3. Comprehensive Model Evaluation

    bic_val = gmm.bic(X)
    aic_val = gmm.aic(X)

    print("       GMM MODEL SELECTION METRICS       ")   
    print(f"BIC (Bayesian Info Criterion, Lower better)  : {bic_val:.4f}")
    print(f"AIC (Akaike Info Criterion, Lower better)    : {aic_val:.4f}")
    print("="*45 + "\n")

    # 4. Cluster Profile Summary
    summary = df.groupby('Cluster')[features].mean().reset_index()
    print("Cluster Means:")
    print(summary.to_string(index=False))

    # 5. Save Segmented Dataset and model
    output_filename = 'Data/customers_segmented_gmm.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nSegmented data saved to '{output_filename}'.")

    joblib.dump(gmm,'Models/GMM_model_2.pkl')

    # 6. Visualize Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='Annual Income ($)', 
        y='Spending Score (1-100)', 
        hue='Cluster', 
        palette='viridis', 
        s=60, 
        alpha=0.8
    )
    plt.title('Customer Financial Segmentation (Gaussian Mixture Model)', fontsize=14)
    plt.xlabel('Annual Income (Normalized)', fontsize=12)
    plt.ylabel('Spending Score (Normalized)', fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
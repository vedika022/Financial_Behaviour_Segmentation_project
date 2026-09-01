import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR/ 'Data'/ 'Customers.csv'
df = pd.read_csv(RAW_DATA_PATH)

print("\nDisplaying First Five columns :")
print(df.head())

data_summary = pd.DataFrame({
    "dtype": df.dtypes,
    "missing_values": df.isnull().sum(),
    "missing_percentage": df.isnull().mean() * 100,
    "unique_values": df.nunique()
})

print("\n Display Data Summary")
print(data_summary, '\n')

# Finding Outliers 
features_name = ['Annual Income ($)', 'Spending Score (1-100)', 'Family Size','Work Experience']
features = df[features_name]

outliers_index = []

for column in features_name:
    # Calculate IQR
    Q1 = features[column].quantile(0.25)
    Q3 = features[column].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = features[(features[column] < lower_bound) | (features[column] > upper_bound)]
    
    for idx in outliers.index:
        if idx not in outliers_index:
            outliers_index.append(idx)
    
    print(f"Number of outliers in {column}: {len(outliers)}")
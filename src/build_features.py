import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR/ 'Data'/ 'Customers.csv'
PROCESSED_DATA_PATH = ROOT_DIR / "Data" / "customers_processed.csv"

df = pd.read_csv(RAW_DATA_PATH)

df.fillna({'Profession': df['Profession'].mode()[0] }, inplace=True)

features = df[['Annual Income ($)', 'Spending Score (1-100)']]
scaler = MinMaxScaler()
features_norm = scaler.fit_transform(features)
features_norm = pd.DataFrame(features_norm, columns=features.columns)

# save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save processed data
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
features_norm.to_csv(PROCESSED_DATA_PATH, index=False)


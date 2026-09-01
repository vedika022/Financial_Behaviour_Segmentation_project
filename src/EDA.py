import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR/ 'Data'/ 'Customers.csv'
df = pd.read_csv(RAW_DATA_PATH)

df = pd.read_csv(RAW_DATA_PATH)

def show_numeric_histplots() :

    plt.figure(figsize=(15, 5))

    plt.subplot(1,3,1)
    sns.histplot(df['Age'], kde=True, bins=25)
    plt.title('Age Distribution')

    plt.subplot(1,3,2)
    sns.histplot(df['Annual Income ($)'], kde=True, bins=25)
    plt.title('Annual Income Distribution')

    plt.subplot(1,3,3)
    sns.histplot(df['Spending Score (1-100)'], kde=True, bins=25)
    plt.title('Spending Score Distribution')

    plt.tight_layout()

    plt.show()



def show_catergorical_barplots():

    plt.figure(figsize=(15, 7))

    plt.subplot(1,2,1)
    df['Gender'].value_counts().plot(kind='bar')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count') 

    plt.subplot(1,2,2)
    df["Profession"].value_counts().plot(kind="bar")
    plt.title("Customer Distribution by Profession")
    plt.xlabel("profession")
    plt.ylabel("count")
    plt.xticks(rotation=45)

    plt.show()

def show_other_numeric_data():

    plt.figure(figsize=(15, 5))

    plt.subplot(1,2,1)
    sns.histplot(df['Work Experience'], kde=True, bins=25)
    plt.title('Work Experience Distribution')
    
    plt.subplot(1,2,2)
    sns.histplot(df['Family Size'], kde=True, bins=25)
    plt.title('Family Size Distribution')

    plt.show()

def show_bivariate_plots():

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(
        data=df,
        x="Annual Income ($)",
        y="Spending Score (1-100)"
    )
    plt.title("Income vs Spending Score")

    plt.subplot(1, 3, 2)
    sns.scatterplot(
        data=df,
        x="Age",
        y="Spending Score (1-100)"
    )
    plt.title("Age vs Spending Score")

    plt.subplot(1, 3, 3)
    sns.scatterplot(
        data=df,
        x="Age",
        y="Annual Income ($)"
    )
    plt.title("Age vs Annual Income")

    plt.tight_layout()
    plt.show()

def show_correlation_heatmap():

    numerical_features = [
        "Age",
        "Annual Income ($)",
        "Spending Score (1-100)",
        "Work Experience",
        "Family Size"
    ]

    correlation = df[numerical_features].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def show_numeric_boxplots():

    numerical_features = [
        "Age",
        "Annual Income ($)",
        "Spending Score (1-100)",
        "Work Experience",
        "Family Size"
    ]

    plt.figure(figsize=(12, 8))

    for i, column in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[column])
        plt.title(column)

    plt.tight_layout()
    plt.show()

show_numeric_histplots()
show_catergorical_barplots()
show_other_numeric_data()
show_bivariate_plots()
show_correlation_heatmap()
show_numeric_boxplots()

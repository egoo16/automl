# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import sys
import os

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Separar las variables numéricas y categóricas
    num_cols = df.select_dtypes(include=['float64', 'int']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Normalizar variables numéricas
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Codificación One-Hot de variables categóricas
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop("target_column", axis=1)  # Reemplaza "target_column" con el nombre de tu variable objetivo
    y = df["target_column"]  # Reemplaza "target_column" con el nombre de tu variable objetivo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    filepath = sys.argv[1]
    df = load_data(filepath)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Guardar los conjuntos de entrenamiento y prueba
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

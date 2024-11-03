# src/data_exploration.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

output_dir = "data/exploration_results"
os.makedirs(output_dir, exist_ok=True)

# Cargar el dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Exploración básica de datos
def explore_data(df):
    # Guardar resumen de primeras filas, descripción y valores faltantes en un archivo
    with open(f"{output_dir}/exploration_summary.txt", "w") as f:
        f.write("Exploración de datos\n\n")
        f.write("Primeras filas del dataset:\n")
        f.write(str(df.head()) + "\n\n")
        f.write("Descripción estadística:\n")
        f.write(str(df.describe()) + "\n\n")
        f.write("Valores Nulos:\n")
        f.write(str(df.isnull().sum()) + "\n")

# Visualización de distribuciones
def visualize_data(df):
    # Guardar el histograma para variables numéricas
    num_cols = df.select_dtypes(include=['float64', 'int']).columns
    df[num_cols].hist(bins=15, figsize=(15, 10))
    plt.suptitle('Distribución de variables numéricas')
    plt.savefig(f"{output_dir}/numerical_distributions.png")
    plt.close()

    # Guardar gráfico de conteo para cada variable categórica
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure()
        sns.countplot(y=col, data=df)
        plt.title(f'Distribución de {col}')
        plt.savefig(f"{output_dir}/{col}_distribution.png")
        plt.close()

if __name__ == "__main__":
    filepath = sys.argv[1]
    df = load_data(filepath)
    explore_data(df)
    visualize_data(df)

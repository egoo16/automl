import pandas as pd
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error
import joblib
import sys
import os

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Cargar datos de entrenamiento y prueba
def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

# Entrenar modelo utilizando TPOT
def train_automl(X_train, y_train, X_test, y_test):
    tpot = TPOTRegressor(verbosity=2, random_state=42, generations=5, population_size=20, config_dict='TPOT sparse')
    tpot.fit(X_train, y_train)

    # Predecir y calcular el error
    predictions = tpot.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Guardar el mejor modelo
    tpot.export(f"{output_dir}/best_model.py")
    joblib.dump(tpot.fitted_pipeline_, f"{output_dir}/best_model.joblib")

    return mse

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    mse = train_automl(X_train, y_train, X_test, y_test)

    # Guardar resultados en un archivo de texto
    with open(f"{output_dir}/automl_results.txt", "w") as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Best pipeline exported as: best_model.py\n")

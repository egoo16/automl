import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from tpot import TPOTRegressor

output_dir = "models"

def train_automl(X_train, y_train, X_test, y_test):
    tpot = TPOTRegressor(
        verbosity=2,
        random_state=42,
        generations=5,
        population_size=20,
        config_dict='TPOT sparse',
        cv=5  # validación cruzada
    )
    
    tpot.fit(X_train, y_train)

    # Predecir y calcular el error
    predictions = tpot.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # guardamos el mejor modelo
    tpot.export(f"{output_dir}/best_model.py")
    joblib.dump(tpot.fitted_pipeline_, f"{output_dir}/best_model.joblib")

    return mse

if __name__ == "__main__":
    # cargar datos
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # entrenar y evaluar el modelo
    mse = train_automl(X_train, y_train, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

import optuna
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = pd.read_csv("data/processed/X_train.csv")
y = pd.read_csv("data/processed/y_train.csv").values.ravel()


def objective(trial):
    # Hiperparámetros de optimización
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-3,  
        warm_start=True
    )
    
    # Dividir los datos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    min_mse = float("inf")
    early_stopping_rounds = 10
    no_improvement_count = 0

    # Entrenamiento con early stopping manual
    for i in range(1, n_estimators + 1):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        
        # Validación y control de early stopping
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        
        if mse < min_mse:
            min_mse = mse
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= early_stopping_rounds:
            break

    return min_mse

if __name__ == "__main__":
    # crear el estudio de Optuna y ejecutarlo
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50) 

    # Guardar los mejores hiperparámetros y el modelo entrenado
    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    best_model.fit(X_train, y_train)
    
    joblib.dump(study.best_trial, "results/best_hyperparameters.joblib")

    # Guardar los resultados de Optuna
    with open("results/optuna_best_params.txt", "w") as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best MSE: {study.best_value}\n")

    # Guardar las curvas de optimización
    optuna.visualization.plot_optimization_history(study).write_html("results/optuna_optimization_history.html")
    optuna.visualization.plot_param_importances(study).write_html("results/optuna_param_importance.html")

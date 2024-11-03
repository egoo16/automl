import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def interpret_model(output_file):
    model = joblib.load("models/best_model.joblib")
    
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # predicciones
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with open(output_file, 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'R^2 Score: {r2}\n')

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Identificar características
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Exportar importancias a CSV
    feature_importance_df.to_csv('results/feature_importances.csv', index=False)

    # Exportar métricas a CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'R^2 Score'],
        'Value': [mse, r2]
    })
    metrics_df.to_csv('results/metrics.csv', index=False)

    # Exportar métricas a Markdown
    with open('results/metrics.md', 'w') as md_file:
        md_file.write('# Model Evaluation Metrics\n')
        md_file.write('| Metric | Value |\n')
        md_file.write('|--------|-------|\n')
        md_file.write(f'| MSE | {mse} |\n')
        md_file.write(f'| R^2 Score | {r2} |\n')


if __name__ == "__main__":
    output_file = "results/interpretation_results.txt"
    interpret_model(output_file)

import os
import sys
import traceback
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Configuración de Rutas ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

dataset_path = os.path.join(root_dir, 'job_salary_prediction_dataset.csv')
model_path = os.path.join(root_dir, 'model.pkl')
mlruns_dir = os.path.join(root_dir, 'mlruns')

os.makedirs(mlruns_dir, exist_ok=True)
tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "Salary-Prediction-CI"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=Path(mlruns_dir).as_uri())
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# --- Carga de Datos ---
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"--- ERROR: No se encontró el dataset en {dataset_path} ---")
    sys.exit(1)

X = df.drop(columns=['salary'])
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Transformación (Pipeline) ---
categorical_features = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
numerical_features = ['experience_years', 'skills_count', 'certifications']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Parámetros del modelo (puedes ajustarlos para experimentar)
model_params = {"fit_intercept": True}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression(**model_params))
])

# --- Entrenamiento ---
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Registro con MLflow ---
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # 1. Registrar Parámetros
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_features", X.shape[1])

        # 2. Registrar Métricas
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", mse**0.5)

        # 3. Inferir Firma (Signature) y Ejemplo de Entrada
        # La firma detecta los tipos de datos de las columnas de entrada y la salida
        signature = infer_signature(X_train, pipeline.predict(X_train.head(5)))
        
        # Tomamos la primera fila como ejemplo representativo
        input_example = X_train.iloc[[0]]

        # 4. Registrar el modelo como artefacto estructurado
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model_salary",
            signature=signature,
            input_example=input_example
        )

        # Guardado físico local para el paso de validación del CI
        joblib.dump(pipeline, model_path)
        
        print(f"✅ Ejecución {run.info.run_id} completada.")
        print(f"✅ MSE: {mse:.4f} | Firma y ejemplo registrados en MLflow.")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
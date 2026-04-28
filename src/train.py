import os
import sys
import traceback
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# La ejecución desde el Makefile asegura que os.getcwd() sea la raíz del proyecto
workspace_dir = os.getcwd() 
mlruns_dir = os.path.join(workspace_dir, "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)

tracking_uri = "sqlite:///mlflow.db"
artifact_location = Path(mlruns_dir).as_uri()

mlflow.set_tracking_uri(tracking_uri)

experiment_name = "Salary-Prediction-CI"
experiment_id = None 

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location 
    )
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            sys.exit(1)
    else:
        raise e 

# --- Carga de Datos ---
try:
    df = pd.read_csv('job_salary_prediction_dataset.csv')
except FileNotFoundError:
    print("--- ERROR: No se encontró el dataset en la raíz ---")
    sys.exit(1)

X = df.drop(columns=['salary'])
y = df['salary']

# --- Transformación (Pipeline) ---
categorical_features = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
numerical_features = ['experience_years', 'skills_count', 'certifications']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entrenamiento ---
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Registro ---
try:
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")
        joblib.dump(pipeline, "model.pkl")
        print(f"✅ Modelo entrenado y guardado en la raíz. MSE: {mse:.4f}")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
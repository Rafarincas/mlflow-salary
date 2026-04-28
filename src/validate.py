import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

THRESHOLD = 80000000.0  

try:
    df = pd.read_csv('job_salary_prediction_dataset.csv')
except FileNotFoundError:
    print("❌ ERROR: No se encontró el dataset en la raíz.")
    sys.exit(1)

X = df.drop(columns=['salary'])
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# os.getcwd() apunta a la raíz, donde train.py dejó el modelo
model_path = os.path.abspath(os.path.join(os.getcwd(), "model.pkl"))

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'model.pkl' en la raíz.")
    sys.exit(1)  

y_pred = model.predict(X_test)  
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)  
else:
    print("❌ El modelo no cumple el umbral.")
    sys.exit(1)
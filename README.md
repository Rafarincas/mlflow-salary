# Pipeline de MLOps: Predicción de Salarios Industriales

Este proyecto implementa un ciclo de vida completo de Machine Learning (MLOps) para predecir salarios basados en perfiles profesionales. El enfoque principal no es solo la precisión del modelo, sino la **reproducibilidad**, el **gobierno de modelos** y la **automatización de la calidad** mediante Integración Continua (CI).

## 📌 Descripción del Proyecto
El objetivo es determinar el salario estimado de un empleado basado en variables demográficas y profesionales (experiencia, educación, industria, etc.). Se utiliza un modelo de Regresión Lineal encapsulado en un `Pipeline` de Scikit-Learn para garantizar la consistencia en la transformación de datos.

## 🛠️ Arquitectura Técnica

### 1. Preprocesamiento y Modelo
Para manejar la naturaleza heterogénea de los datos, se implementó un `ColumnTransformer`:
* **Variables Numéricas:** Escaladas mediante `StandardScaler` para normalizar la influencia de magnitudes (ej. años de experiencia vs cantidad de habilidades).
* **Variables Categóricas:** Transformadas mediante `OneHotEncoder` para permitir que el modelo matemático interprete variables de texto (ej. industria o nivel educativo).
* **Pipeline:** El modelo y los preprocesadores viajan juntos en un solo objeto `model.pkl`. Esto evita el *training-serving skew* (diferencia entre entrenamiento y producción).

### 2. Gobierno con MLflow
Cada ejecución del modelo se registra en un servidor de tracking local (`mlflow.db`) capturando:
* **Parámetros:** Configuración del modelo y del split de datos.
* **Métricas:** Error Cuadrático Medio (MSE) y RMSE.
* **Firmas (Signatures):** Definición estricta de los tipos de datos de entrada y salida, actuando como un contrato para futuras integraciones.
* **Artefactos:** El modelo empaquetado junto con un ejemplo de entrada (`input_example.json`).

### 3. Integración Continua (GitHub Actions)
Se implementó un flujo de CI automatizado en Ubuntu que se dispara con cada `git push` a la rama `main`:
1.  **Entorno:** Instalación automática de dependencias mediante `requirements.txt`.
2.  **Entrenamiento:** Ejecución de `src/train.py` para generar un nuevo modelo fresco con los datos más recientes.
3.  **Validación de Calidad:** Ejecución de `src/validate.py`. Si el MSE del nuevo modelo supera el umbral de calidad definido ($80,000,000$), el pipeline se detiene y marca un error, bloqueando el paso a producción de un modelo deficiente.
4.  **Entrega:** Si el modelo es apto, se genera un artefacto descargable en GitHub llamado `modelo-salario-aprobado`.

## 📂 Estructura del Repositorio
```text
├── .github/workflows/  # Configuración de la automatización (CI)
├── src/                # Código fuente (Lógica de ML)
│   ├── train.py        # Entrenamiento y registro en MLflow
│   └── validate.py     # Pruebas de calidad automáticas
├── Makefile            # Abstracción de comandos para el servidor CI
├── requirements.txt    # Dependencias del proyecto
├── model.pkl           # El modelo validado (generado localmente)
└── job_salary_prediction_dataset.csv # Datos de entrenamiento

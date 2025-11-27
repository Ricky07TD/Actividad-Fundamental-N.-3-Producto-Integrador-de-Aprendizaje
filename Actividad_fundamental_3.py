import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración visual
sns.set(style="whitegrid")


# 1. CARGA DE DATOS (Dataset Público)

# Usamos la URL raw de GitHub para no depender de archivos locales
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
print("Cargando dataset desde:", url)
df = pd.read_csv(url)

# Exploración inicial
print(f"\nDimensiones del dataset: {df.shape}")
print(df.head())


# 2. LIMPIEZA DE DATOS

# Revisar nulos
print("\n Revisión de valores nulos ")
print(df.isnull().sum())

# Revisar y eliminar duplicados
duplicados = df.duplicated().sum()
if duplicados > 0:
    print(f"\nEliminando {duplicados} registros duplicados...")
    df = df.drop_duplicates()
else:
    print("\nNo se encontraron duplicados.")


# 3. PREPROCESAMIENTO (Encoding y Scaling)

# Separamos las variables predictoras (X) de la variable objetivo (y)
X = df.drop('charges', axis=1)  # Características
y = df['charges']               # Lo que queremos predecir (Costo)

# Identificamos columnas numéricas y categóricas
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Creamos el "Transformer" que hará todo el trabajo pesado:
# 1. StandardScaler para normalizar números (Edad, BMI)
# 2. OneHotEncoder para convertir texto a números (Sexo, Fumador, Región)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Aplicamos las transformaciones a X
X_processed = preprocessor.fit_transform(X)

print("\n Preprocesamiento completado ")
print("Ejemplo de datos transformados (primera fila):")
print(X_processed[0])


# 4. DIVISIÓN DE DATOS (TRAIN / TEST)

# Usamos 70% para entrenar y 30% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42
)

print(f"\nDatos de Entrenamiento: {X_train.shape[0]} registros")
print(f"Datos de Prueba: {X_test.shape[0]} registros")


# 5. IMPLEMENTACIÓN DEL MODELO (REGRESIÓN LINEAL)

model = LinearRegression()
model.fit(X_train, y_train)  # Entrenamos con el 70%

# Hacemos predicciones con el 30% restante para validar
y_pred = model.predict(X_test)

print("\n Modelo Entrenado ")
print(f"Intercepto (Bias): {model.intercept_:.2f}")


# 6. EVALUACIÓN Y MÉTRICAS

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nResultados de la Evaluacion:")
print(f"R2 Score (Precision): {r2:.4f}")
print(f"MAE (Error Medio Absoluto): {mae:.2f}")
print(f"MSE (Error Cuadratico Medio): {mse:.2f}")


# 7. GRÁFICAS

plt.figure(figsize=(10, 6))

# Gráfico de dispersión: Real vs Predicho
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue', label='Predicciones')

# Línea de referencia perfecta
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color='red', lw=2, linestyle='--', label='Prediccion Perfecta')

plt.title("Evaluacion del Modelo: Valores Reales vs Predichos")
plt.xlabel("Costo Real ($)")
plt.ylabel("Costo Predicho ($)")
plt.legend()

plt.savefig('grafica_prediccion.png', dpi=300)
plt.show()
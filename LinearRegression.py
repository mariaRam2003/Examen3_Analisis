from sklearn.linear_model import LinearRegression
import numpy as np

# Ejemplo de datos de entrenamiento
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# Crear y entrenar el modelo de regresión lineal
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Coeficientes de la línea de regresión
print("Coeficiente:", regression_model.coef_)
print("Intercepto:", regression_model.intercept_)

# Predecir con nuevos datos
X_new = np.array([[6]])
y_pred = regression_model.predict(X_new)
print("Predicción:", y_pred)
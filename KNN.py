from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Ejemplo de datos de entrenamiento
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# Crear y entrenar el modelo KNN
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predecir con nuevos datos
X_new = np.array([[6]])
y_pred = knn_model.predict(X_new)
print("Predicción:", y_pred)

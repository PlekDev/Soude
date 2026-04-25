import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# 1. Leer el libro de texto (tu archivo CSV)
print("Cargando los datos...")
df = pd.read_csv('dataset_entrenamiento.csv')

X = df[['Energia_C3', 'Energia_C4']] # Las características (los números)
y = df['Clase'] # Las respuestas (Reposo o Passthought)

# 2. El alumno (La Inteligencia Artificial) empieza a estudiar
print("Entrenando la Inteligencia Artificial...")
modelo_svm = SVC(kernel='linear')
modelo_svm.fit(X, y)

# 3. Le hacemos un examen rápido para ver si aprendió
predicciones = modelo_svm.predict(X)
precision = accuracy_score(y, predicciones) * 100
print(f"🎯 Examen aprobado con: {precision}% de precisión")

# 4. Guardamos el cerebro del alumno en un archivo
with open('modelo_passthought.pkl', 'wb') as archivo:
    pickle.dump(modelo_svm, archivo)

print("✅ ¡Cerebro guardado exitosamente como 'modelo_passthought.pkl'!")
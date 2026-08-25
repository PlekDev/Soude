"""
ml/train_model.py — entrena el SVM Rest/Passthought sobre data/dataset_phase2.csv.

Reporta accuracy honesta (split 70/30 y validación cruzada 5-fold — nunca sobre
el mismo set de entrenamiento) y guarda {'model', 'scaler'} en
data/passthought_model.pkl, que es exactamente lo que carga ml/test_model.py.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASET_CSV = ROOT / "data" / "dataset_phase2.csv"
MODEL_PKL   = ROOT / "data" / "passthought_model.pkl"

FEATURES = ["Variance_C3", "Variance_C4"]
LABEL    = "Class"


def main() -> None:
    if not DATASET_CSV.exists():
        raise SystemExit(f"No existe {DATASET_CSV}. Corre primero ml/create_database.py")
    print(f"Cargando {DATASET_CSV}...")
    df = pd.read_csv(DATASET_CSV)
    X = df[FEATURES].to_numpy()
    y = df[LABEL].to_numpy()

    # Accuracy honesta 1: split estratificado, el modelo nunca ve el set de prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    model  = SVC(kernel="linear").fit(scaler.transform(X_train), y_train)
    test_acc = accuracy_score(y_test, model.predict(scaler.transform(X_test))) * 100

    # Accuracy honesta 2: validacion cruzada 5-fold (el scaler se ajusta por fold
    # dentro del pipeline, sin fuga de informacion)
    cv_pipe   = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    cv_scores = cross_val_score(cv_pipe, X, y, cv=5)

    print(f"Accuracy (split 70/30, datos no vistos):  {test_acc:.1f}%")
    print(f"Accuracy (validacion cruzada 5-fold):     "
          f"{cv_scores.mean() * 100:.1f}% +/- {cv_scores.std() * 100:.1f}%")

    # Modelo final: re-entrenado con TODO el dataset para produccion
    final_scaler = StandardScaler().fit(X)
    final_model  = SVC(kernel="linear").fit(final_scaler.transform(X), y)

    MODEL_PKL.parent.mkdir(exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": final_model, "scaler": final_scaler}, f)
    print(f"Modelo guardado en {MODEL_PKL}")


if __name__ == "__main__":
    main()

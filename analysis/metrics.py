"""
analysis/metrics.py — métricas de desempeño biométrico para Soude.

Dos niveles de análisis:

1. Punto de operación real (el que usa el sistema hoy en producción):
   usa la columna `granted` tal cual la decidió signal_processing.py
   (regla: delta_uv >= 1.5 µV AND delta_uv >= 1.5 * pre_sigma_uv).
       FAR = % de sesiones impostor que fueron GRANTED (aceptadas por error)
       FRR = % de sesiones genuine que fueron DENIED (rechazadas por error)

2. Curva ROC / EER barriendo el umbral sobre `delta_uv` como score de
   decisión, para poder reportar "a qué EER podría operar el sistema"
   independientemente del umbral fijo actual — esto es lo que se suele
   pedir en papers de biometría EEG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def operating_point_metrics(df: pd.DataFrame) -> dict:
    """FAR/FRR con la decisión real del sistema (columna `granted`)."""
    genuine = df[df["session_type"] == "genuine"]
    impostor = df[df["session_type"] == "impostor"]

    if len(genuine) == 0 or len(impostor) == 0:
        raise SystemExit(
            "Necesito al menos una sesión genuine y una impostor etiquetadas "
            "en labels.csv para calcular FAR/FRR."
        )

    far = (impostor["granted"] == True).mean()   # noqa: E712
    frr = (genuine["granted"] == False).mean()    # noqa: E712

    return {
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "n_genuine_granted": int((genuine["granted"] == True).sum()),   # noqa: E712
        "n_genuine_denied": int((genuine["granted"] == False).sum()),   # noqa: E712
        "n_impostor_granted": int((impostor["granted"] == True).sum()), # noqa: E712 (falsos aceptados)
        "n_impostor_denied": int((impostor["granted"] == False).sum()), # noqa: E712
        "FAR": far,
        "FRR": frr,
        "accuracy": 1.0 - (far * len(impostor) + frr * len(genuine)) / (len(genuine) + len(impostor)),
    }


def roc_and_eer(df: pd.DataFrame, score_col: str = "delta_uv") -> dict:
    """
    Barre el umbral sobre `score_col` (por defecto delta_uv, la separación
    target/non-target en µV) y calcula FAR(t)/FRR(t) para toda la curva.

    Convención: acceso se otorga si score >= threshold (igual que la regla
    de amplitud del sistema real).

    Retorna thresholds, far_curve, frr_curve, eer, eer_threshold, auc.
    """
    genuine_scores = df.loc[df["session_type"] == "genuine", score_col].dropna().to_numpy()
    impostor_scores = df.loc[df["session_type"] == "impostor", score_col].dropna().to_numpy()

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise SystemExit(f"Faltan valores de {score_col} en genuine o impostor.")

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.unique(np.concatenate([all_scores, [all_scores.min() - 1e-6, all_scores.max() + 1e-6]]))
    thresholds.sort()

    far_curve = np.array([(impostor_scores >= t).mean() for t in thresholds])   # false accepts
    frr_curve = np.array([(genuine_scores < t).mean() for t in thresholds])     # false rejects
    tpr_curve = 1.0 - frr_curve  # true accepts (genuine correctly granted)

    # EER: punto donde FAR y FRR se cruzan (más cercano)
    diff = np.abs(far_curve - frr_curve)
    eer_idx = int(np.argmin(diff))
    eer = (far_curve[eer_idx] + frr_curve[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    # AUC por regla del trapecio sobre (FAR, TPR) ordenado por FAR ascendente
    order = np.argsort(far_curve)
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renombró trapz
    auc = float(_trapz(tpr_curve[order], far_curve[order]))

    return {
        "score_col": score_col,
        "thresholds": thresholds,
        "far_curve": far_curve,
        "frr_curve": frr_curve,
        "tpr_curve": tpr_curve,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores),
    }


def print_report(df: pd.DataFrame) -> None:
    op = operating_point_metrics(df)
    print("=" * 60)
    print("PUNTO DE OPERACIÓN ACTUAL (umbral fijo del sistema)")
    print("=" * 60)
    print(f"Sesiones genuine:  {op['n_genuine']}  "
          f"(granted={op['n_genuine_granted']}, denied={op['n_genuine_denied']})")
    print(f"Sesiones impostor: {op['n_impostor']}  "
          f"(granted={op['n_impostor_granted']}, denied={op['n_impostor_denied']})")
    print(f"FAR (impostores aceptados): {op['FAR']*100:.2f}%")
    print(f"FRR (genuinos rechazados):  {op['FRR']*100:.2f}%")
    print(f"Accuracy global:            {op['accuracy']*100:.2f}%")

    print()
    print("=" * 60)
    print("ANÁLISIS ROC / EER (barriendo umbral sobre delta_uv)")
    print("=" * 60)
    roc = roc_and_eer(df)
    print(f"EER: {roc['eer']*100:.2f}%  (en threshold delta_uv = {roc['eer_threshold']:.3f} µV)")
    print(f"AUC: {roc['auc']:.4f}")
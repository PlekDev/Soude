"""
analysis/erp_analysis.py — grand-average del ERP (target vs non-target)
por grupo (genuine/impostor), usando el `erp_data` ya calculado por
signal_processing.py y guardado en cada auth_result.json.

No necesita tocar epochs.npz: erp_data.t_ms / target / nontarget ya son
las formas de onda promedio por sesión, listas para promediar entre
sesiones (grand average).

Uso:
    from analysis.erp_analysis import grand_average
    ga = grand_average(df, group="genuine")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_erp_data(session_dir: str) -> dict | None:
    auth_path = Path(session_dir) / "auth_result.json"
    with open(auth_path, encoding="utf-8") as f:
        auth = json.load(f)
    return auth.get("erp_data")


def grand_average(df: pd.DataFrame, group: str) -> dict:
    """
    Promedia las curvas ERP (target y non-target) de todas las sesiones
    de un grupo ('genuine' o 'impostor'). Requiere que todas las sesiones
    compartan el mismo eje t_ms (mismo epoch window / sample rate); si no,
    interpola al eje de la primera sesión válida.
    """
    subset = df[(df["session_type"] == group) & (df["has_erp_data"])]
    if len(subset) == 0:
        raise SystemExit(f"No hay sesiones con erp_data para el grupo '{group}'.")

    t_ref = None
    target_curves = []
    nontarget_curves = []

    for session_dir in subset["session_dir"]:
        erp = _load_erp_data(session_dir)
        if not erp or not erp.get("t_ms"):
            continue
        t = np.asarray(erp["t_ms"], dtype=float)
        target = np.asarray(erp["target"], dtype=float)
        nontarget = np.asarray(erp["nontarget"], dtype=float)

        if t_ref is None:
            t_ref = t
        elif not np.array_equal(t, t_ref):
            target = np.interp(t_ref, t, target)
            nontarget = np.interp(t_ref, t, nontarget)

        target_curves.append(target)
        nontarget_curves.append(nontarget)

    if t_ref is None or not target_curves:
        raise SystemExit(f"Ninguna sesión de '{group}' tenía erp_data con contenido.")

    target_curves = np.vstack(target_curves)
    nontarget_curves = np.vstack(nontarget_curves)

    return {
        "group": group,
        "n_sessions": len(target_curves),
        "t_ms": t_ref,
        "target_mean": target_curves.mean(axis=0),
        "target_sem": target_curves.std(axis=0, ddof=1) / np.sqrt(len(target_curves)),
        "nontarget_mean": nontarget_curves.mean(axis=0),
        "nontarget_sem": nontarget_curves.std(axis=0, ddof=1) / np.sqrt(len(nontarget_curves)),
    }
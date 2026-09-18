"""
analysis/load_sessions.py — carga todas las sesiones de logs/ + labels.csv
en un pandas.DataFrame listo para análisis.

Cada fila = una sesión, con las métricas estructuradas de auth_result.json
más session_type (desde labels.csv o, si existe, desde el propio json).

Uso como librería:
    from analysis.load_sessions import load_all_sessions
    df = load_all_sessions("logs", "labels.csv")

Uso como script (imprime resumen):
    python analysis/load_sessions.py --logs-dir logs --labels labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_labels(labels_path: Path) -> dict[str, str]:
    if not labels_path.exists():
        return {}
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["session_id"]: row["session_type"].strip().lower()
            for row in reader
            if row.get("session_type", "").strip().lower() in ("genuine", "impostor")
        }


def _load_one_session(session_dir: Path, labels: dict[str, str]) -> dict | None:
    auth_path = session_dir / "auth_result.json"
    if not auth_path.exists():
        return None

    with open(auth_path, encoding="utf-8") as f:
        auth = json.load(f)

    sid = session_dir.name

    # session_type: prioriza labels.csv (etiqueta manual del usuario) sobre
    # el campo guardado en el propio json. Esto es a propósito: el sistema
    # solo guarda "impostor" si se corrió con SOUDE_SESSION_TYPE=impostor;
    # si esa variable nunca se usó, TODAS las sesiones quedaron marcadas
    # "genuine" en el json aunque muchas hayan sido impostor en la práctica.
    # labels.csv es la fuente de verdad; si una sesión no aparece ahí, se
    # cae al campo del json como último recurso.
    session_type = labels.get(sid) or auth.get("session_type")

    n_markers = None
    n_target = auth.get("n_target")
    n_nontarget = auth.get("n_nontarget")
    markers_path = session_dir / "markers.csv"
    if markers_path.exists():
        mdf = pd.read_csv(markers_path)
        n_markers = len(mdf)
        if n_target is None:
            n_target = int(mdf["is_target"].sum())
        if n_nontarget is None:
            n_nontarget = int((mdf["is_target"] == 0).sum())

    target_peak = auth.get("target_peak_uv")
    nontarget_peak = auth.get("nontarget_peak_uv")
    # delta_uv puede no venir guardado en json's viejos; se deriva igual
    # que en signal_processing.py: diferencia absoluta target vs non-target.
    delta_uv = auth.get("delta_uv")
    if delta_uv is None and target_peak is not None and nontarget_peak is not None:
        delta_uv = abs(target_peak - nontarget_peak)

    row = {
        "session_id": sid,
        "session_type": session_type,
        "granted": bool(auth.get("granted")),
        "target_peak_uv": target_peak,
        "nontarget_peak_uv": nontarget_peak,
        "delta_uv": delta_uv,
        "pre_sigma_uv": auth.get("pre_sigma_uv"),
        "snr_db": auth.get("snr_db"),
        "n_target": n_target,
        "n_nontarget": n_nontarget,
        "n_markers": n_markers,
        "message": auth.get("message"),
        "has_epochs": (session_dir / "epochs.npz").exists(),
        "has_erp_data": bool(auth.get("erp_data")),
        "session_dir": str(session_dir),
    }
    return row


def load_all_sessions(logs_dir: str | Path = "logs",
                       labels_path: str | Path = "labels.csv") -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    labels_path = Path(labels_path)
    if not logs_dir.exists():
        raise SystemExit(f"No existe la carpeta {logs_dir}")

    labels = _load_labels(labels_path)

    rows = []
    for session_dir in sorted(logs_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        row = _load_one_session(session_dir, labels)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No se encontraron sesiones válidas en {logs_dir}")

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs", type=Path)
    ap.add_argument("--labels", default="labels.csv", type=Path)
    args = ap.parse_args()

    df = load_all_sessions(args.logs_dir, args.labels)

    print(f"Total de sesiones encontradas: {len(df)}")
    print(df["session_type"].value_counts(dropna=False).to_string())
    n_unlabeled = df["session_type"].isna().sum()
    if n_unlabeled:
        print(f"\n⚠️  {n_unlabeled} sesiones sin etiquetar (session_type vacío).")
        print("   Llena labels.csv (corre make_labels_template.py si no existe)")
        print("   antes de calcular FAR/FRR.")
    print("\nPrimeras filas:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
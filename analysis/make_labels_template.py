"""
analysis/make_labels_template.py — genera (o actualiza) labels.csv a partir
de las carpetas de sesión en logs/.

Uso:
    python analysis/make_labels_template.py [--logs-dir logs] [--out labels.csv]

Crea un CSV con las columnas:
    session_id, session_type, notes

Si labels.csv ya existe, conserva las etiquetas que ya hayas llenado y solo
agrega las carpetas nuevas que aún no estén en el archivo (no pisa tu trabajo).

session_type debe llenarse manualmente con "genuine" o "impostor" para cada
fila. Cualquier otro valor (vacío, "?", etc.) se trata como "sin etiquetar"
y esas sesiones se excluyen del cálculo de FAR/FRR más adelante.
"""
import argparse
import csv
from pathlib import Path


def find_sessions(logs_dir: Path) -> list[str]:
    if not logs_dir.exists():
        raise SystemExit(f"No existe la carpeta {logs_dir}")
    sessions = sorted(
        p.name for p in logs_dir.iterdir()
        if p.is_dir() and (p / "auth_result.json").exists()
    )
    if not sessions:
        raise SystemExit(f"No encontré carpetas de sesión (con auth_result.json) en {logs_dir}")
    return sessions


def load_existing(out_path: Path) -> dict[str, dict]:
    if not out_path.exists():
        return {}
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["session_id"]: row for row in reader}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs", type=Path)
    ap.add_argument("--out", default="labels.csv", type=Path)
    args = ap.parse_args()

    sessions = find_sessions(args.logs_dir)
    existing = load_existing(args.out)

    rows = []
    n_new = 0
    for sid in sessions:
        if sid in existing:
            rows.append(existing[sid])
        else:
            rows.append({"session_id": sid, "session_type": "", "notes": ""})
            n_new += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "session_type", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    n_labeled = sum(1 for r in rows if r["session_type"] in ("genuine", "impostor"))
    print(f"{args.out}: {len(rows)} sesiones totales, {n_new} nuevas, {n_labeled} ya etiquetadas.")
    print("Abre el CSV y llena la columna session_type con 'genuine' o 'impostor' para cada fila.")


if __name__ == "__main__":
    main()
"""
analysis/run_analysis.py — corre el análisis completo de las sesiones de
Soude y deja tablas + figuras listas para el paper/congreso.

Uso:
    python analysis/run_analysis.py --logs-dir logs --labels labels.csv --out results

Genera en --out:
    sessions_table.csv        — una fila por sesión, todas las métricas
    operating_point.txt       — FAR/FRR/accuracy con el umbral real del sistema
    roc_curve.png             — curva ROC (FAR vs TPR) + punto de EER
    far_frr_vs_threshold.png  — FAR y FRR en función del umbral, con el EER marcado
    grand_average_erp.png     — ERP promedio target vs non-target, genuine vs impostor
    delta_uv_distribution.png — histograma de separación target/non-target por grupo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from load_sessions import load_all_sessions
from metrics import operating_point_metrics, roc_and_eer
from erp_analysis import grand_average


def plot_roc(roc: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    order = roc["far_curve"].argsort()
    ax.plot(roc["far_curve"][order], roc["tpr_curve"][order], lw=2, label=f"ROC (AUC={roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Azar")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TPR = 1 - FRR")
    ax.set_title("Curva ROC — autenticación P300")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_far_frr_vs_threshold(roc: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(roc["thresholds"], roc["far_curve"] * 100, label="FAR (%)")
    ax.plot(roc["thresholds"], roc["frr_curve"] * 100, label="FRR (%)")
    ax.axvline(roc["eer_threshold"], color="gray", ls="--", lw=1,
               label=f"EER threshold = {roc['eer_threshold']:.2f} µV")
    ax.axhline(roc["eer"] * 100, color="gray", ls=":", lw=1)
    ax.set_xlabel("Umbral de decisión (delta_uv, µV)")
    ax.set_ylabel("Tasa de error (%)")
    ax.set_title(f"FAR / FRR vs. umbral — EER = {roc['eer']*100:.2f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_grand_average(df, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, group in zip(axes, ["genuine", "impostor"]):
        try:
            ga = grand_average(df, group)
        except SystemExit as e:
            ax.set_title(f"{group}: {e}")
            continue
        t = ga["t_ms"]
        ax.plot(t, ga["target_mean"], label="Target", color="crimson")
        ax.fill_between(t, ga["target_mean"] - ga["target_sem"], ga["target_mean"] + ga["target_sem"],
                         color="crimson", alpha=0.2)
        ax.plot(t, ga["nontarget_mean"], label="Non-target", color="steelblue")
        ax.fill_between(t, ga["nontarget_mean"] - ga["nontarget_sem"], ga["nontarget_mean"] + ga["nontarget_sem"],
                         color="steelblue", alpha=0.2)
        ax.axvspan(250, 500, color="gray", alpha=0.1, label="Ventana P300")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{group.capitalize()} (n={ga['n_sessions']} sesiones)")
        ax.set_xlabel("Tiempo (ms)")
    axes[0].set_ylabel("Amplitud (µV)")
    axes[0].legend()
    fig.suptitle("Grand-average ERP: Target vs Non-target")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta_distribution(df, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for group, color in [("genuine", "seagreen"), ("impostor", "indianred")]:
        vals = df.loc[df["session_type"] == group, "delta_uv"].dropna()
        ax.hist(vals, bins=15, alpha=0.5, label=f"{group} (n={len(vals)})", color=color)
    ax.set_xlabel("delta_uv = |Target P300 − Non-target P300| (µV)")
    ax.set_ylabel("Número de sesiones")
    ax.set_title("Distribución de separación target/non-target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs", type=Path)
    ap.add_argument("--labels", default="labels.csv", type=Path)
    ap.add_argument("--out", default="results", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_all_sessions(args.logs_dir, args.labels)
    df.to_csv(args.out / "sessions_table.csv", index=False)
    print(f"Tabla de sesiones -> {args.out / 'sessions_table.csv'}")

    df_labeled = df.dropna(subset=["session_type"])
    n_unlabeled = len(df) - len(df_labeled)
    if n_unlabeled:
        print(f"⚠️  {n_unlabeled} sesiones sin etiquetar, se excluyen de FAR/FRR/ROC/ERP.")

    op = operating_point_metrics(df_labeled)
    roc = roc_and_eer(df_labeled)

    with open(args.out / "operating_point.txt", "w", encoding="utf-8") as f:
        f.write("PUNTO DE OPERACIÓN ACTUAL (umbral fijo del sistema)\n")
        f.write(f"Sesiones genuine:  {op['n_genuine']} (granted={op['n_genuine_granted']}, "
                f"denied={op['n_genuine_denied']})\n")
        f.write(f"Sesiones impostor: {op['n_impostor']} (granted={op['n_impostor_granted']}, "
                f"denied={op['n_impostor_denied']})\n")
        f.write(f"FAR: {op['FAR']*100:.2f}%\n")
        f.write(f"FRR: {op['FRR']*100:.2f}%\n")
        f.write(f"Accuracy: {op['accuracy']*100:.2f}%\n\n")
        f.write(f"EER (barriendo umbral sobre delta_uv): {roc['eer']*100:.2f}% "
                f"@ threshold={roc['eer_threshold']:.3f} µV\n")
        f.write(f"AUC: {roc['auc']:.4f}\n")
    print(f"Métricas -> {args.out / 'operating_point.txt'}")

    plot_roc(roc, args.out / "roc_curve.png")
    plot_far_frr_vs_threshold(roc, args.out / "far_frr_vs_threshold.png")
    plot_delta_distribution(df_labeled, args.out / "delta_uv_distribution.png")
    plot_grand_average(df_labeled, args.out / "grand_average_erp.png")
    print(f"Figuras -> {args.out}/*.png")

    print("\nListo. Revisa la carpeta", args.out)


if __name__ == "__main__":
    main()
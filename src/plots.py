import math
import os

import numpy as np
from scipy import stats

from .config import FIG_DIR, FIG_DPI, MPL_CONFIG_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def _comma_formatter(x, p):
    return f"{x}".replace(".", ",")

def plot_qq_pre_post(df):
    pre = df.loc[df["I"] == 0, "volume_total"].to_numpy(dtype=float)
    post = df.loc[df["I"] == 1, "volume_total"].to_numpy(dtype=float)

    # Pré
    plt.figure(figsize=(7, 6))
    stats.probplot(pre, dist="norm", plot=plt)
    plt.title("Gráfico Q–Q — Pré-Ecstrator (Volume Mensal)", fontsize=14, fontweight="bold")
    plt.xlabel("Quantis Teóricos", fontsize=12)
    plt.ylabel("Valores Ordenados Observados", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "qq_pre_volume.png", dpi=FIG_DPI)
    plt.close()

    # Pós
    plt.figure(figsize=(7, 6))
    stats.probplot(post, dist="norm", plot=plt)
    plt.title("Gráfico Q–Q — Pós-Ecstrator (Volume Mensal)", fontsize=14, fontweight="bold")
    plt.xlabel("Quantis Teóricos", fontsize=12)
    plt.ylabel("Valores Ordenados Observados", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "qq_pos_volume.png", dpi=FIG_DPI)
    plt.close()

def plot_pre_post_sla_bars(df):
    pre = df[df["I"] == 0]
    post = df[df["I"] == 1]

    S_pre, N_pre = int(pre["sucesso_72h"].sum()), int(pre["volume_total"].sum())
    S_post, N_post = int(post["sucesso_72h"].sum()), int(post["volume_total"].sum())

    p1 = S_pre / N_pre
    p2 = S_post / N_post

    labels = ["Pré", "Pós"]
    props = np.array([p1, p2])

    cis = np.array([
        [p1 - 1.96 * math.sqrt(p1 * (1 - p1) / N_pre), p1 + 1.96 * math.sqrt(p1 * (1 - p1) / N_pre)],
        [p2 - 1.96 * math.sqrt(p2 * (1 - p2) / N_post), p2 + 1.96 * math.sqrt(p2 * (1 - p2) / N_post)],
    ])
    yerr = np.vstack([props - cis[:, 0], cis[:, 1] - props])

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, props * 100, color=['lightcoral', 'skyblue'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{yval:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=11)

    plt.errorbar(labels, props * 100, yerr=yerr * 100, fmt="none", capsize=6, color="black")
    plt.title("Migrações Concluídas em ≤72h", fontsize=14, fontweight="bold")
    plt.ylabel("Proporção (%)", fontsize=12)
    plt.xlabel("Período", fontsize=12)
    plt.ylim(0, max(props * 100) * 1.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prop_72h_pre_pos.png", dpi=FIG_DPI)
    plt.close()

def plot_its_volume(df, model, X, intervention_t):
    df_plot = df.copy()
    df_plot["y_hat"] = model.predict(X)

    plt.figure(figsize=(10, 4.8))
    plt.plot(df_plot["t"], df_plot["volume_total"], marker="o", linewidth=1.5, label="Observado")
    plt.plot(df_plot["t"], df_plot["y_hat"], linestyle="--", linewidth=2, label="Ajustado (ITS)")
    plt.axvline(intervention_t, linestyle=":", linewidth=2, label="Intervenção (novembro/2022)")
    plt.title("ITS — Volume mensal de migrações (HAC com 6 defasagens)")
    plt.xlabel("Tempo (índice mensal)")
    plt.ylabel("Migrações por mês")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "its_volume_hac6.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def plot_its_logit(df, res, X, intervention_t):
    df_plot = df.copy()
    df_plot["p_obs"] = df_plot["sucesso_72h"] / df_plot["volume_total"]
    df_plot["p_hat"] = res.predict(X)

    plt.figure(figsize=(10, 4.8))
    plt.plot(df_plot["t"], df_plot["p_obs"] * 100, marker="o", label="Observado (% ≤72h)")
    plt.plot(df_plot["t"], df_plot["p_hat"] * 100, linestyle="--", label="Ajustado (ITS logístico)")
    plt.axvline(intervention_t, linestyle=":", linewidth=2, label="Intervenção (novembro/2022)")
    plt.title("ITS Logístico — Proporção de migrações ≤72h (HAC com 6 defasagens)")
    plt.xlabel("Índice temporal (mês)")
    plt.ylabel("Proporção (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "its_logit_72h_hac6.png", dpi=FIG_DPI)
    plt.close()

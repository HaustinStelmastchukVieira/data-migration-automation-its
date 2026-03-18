
from __future__ import annotations

import os

from .config import AnalysisConfig

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint

from .models import linear_predict_ci


def apply_publication_style(config: AnalysisConfig) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(config.mpl_config_dir))
    plt.rcParams.update({
        "figure.dpi": config.fig_dpi,
        "savefig.dpi": config.fig_dpi,
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
    })


def _style_axis(axis) -> None:
    axis.grid(False)
    axis.set_axisbelow(True)


def _format_ptbr_tick(value: float, decimals: int | None = None) -> str:
    if decimals is None and np.isclose(value, round(value)):
        return f"{int(round(value))}"
    if decimals is None:
        decimals = 1
    return f"{value:.{decimals}f}".replace(".", ",")


def _ptbr_number(value: float, decimals: int = 2) -> str:
    return _format_ptbr_tick(value, decimals)


def _comma_decimal_formatter(decimals: int | None = 1) -> FuncFormatter:
    return FuncFormatter(lambda value, _: _format_ptbr_tick(value, decimals))


def _format_month_axis(axis) -> None:
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def _qq_panel(axis, values: np.ndarray, title: str) -> None:
    (theoretical, observed), (slope, intercept, _) = stats.probplot(values, dist="norm")
    ref_x = np.array([theoretical.min(), theoretical.max()])

    axis.scatter(
        theoretical,
        observed,
        s=22,
        facecolors="white",
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )
    axis.plot(ref_x, slope * ref_x + intercept, color="black", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Quantis teóricos")
    axis.set_ylabel("Quantis observados")
    axis.xaxis.set_major_formatter(_comma_decimal_formatter(decimals=1))
    axis.yaxis.set_major_formatter(_comma_decimal_formatter(decimals=None))
    _style_axis(axis)

    if len(values) >= 3:
        shapiro_w, shapiro_p = stats.shapiro(values)
        axis.text(
            0.04,
            0.96,
            f"n = {len(values)}\nW = {_ptbr_number(shapiro_w, 3)}\np = {_ptbr_number(shapiro_p, 3)}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.95},
        )


def plot_qq_volume_panel(df, config: AnalysisConfig) -> None:
    pre = df.loc[df["I"] == 0, "volume_total"].to_numpy(dtype=float)
    post = df.loc[df["I"] == 1, "volume_total"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)
    _qq_panel(axes[0], pre, "Pré-implantação")
    _qq_panel(axes[1], post, "Pós-implantação")
    fig.suptitle("Diagnóstico Q-Q do volume mensal", fontsize=12, fontweight="bold")
    fig.savefig(config.figures_dir / "qq_volume_panel.png", bbox_inches="tight")
    plt.close(fig)


def plot_pre_post_sla(df, config: AnalysisConfig) -> None:
    pre = df.loc[df["I"] == 0]
    post = df.loc[df["I"] == 1]

    s_pre = int(pre["sucesso_72h"].sum())
    n_pre = int(pre["volume_total"].sum())
    s_post = int(post["sucesso_72h"].sum())
    n_post = int(post["volume_total"].sum())

    proportions = np.array([s_pre / n_pre, s_post / n_post])
    ci_low = np.array([
        proportion_confint(s_pre, n_pre, alpha=0.05, method="wilson")[0],
        proportion_confint(s_post, n_post, alpha=0.05, method="wilson")[0],
    ])
    ci_high = np.array([
        proportion_confint(s_pre, n_pre, alpha=0.05, method="wilson")[1],
        proportion_confint(s_post, n_post, alpha=0.05, method="wilson")[1],
    ])

    x_values = np.array([0.0, 0.62])

    fig, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    axis.plot(x_values, proportions * 100, color="#6f7b8a", linewidth=1.4, zorder=1)
    axis.errorbar(
        x_values,
        proportions * 100,
        yerr=np.vstack([(proportions - ci_low) * 100, (ci_high - proportions) * 100]),
        fmt="none",
        ecolor="#4d4d4d",
        elinewidth=1.8,
        capsize=0,
        zorder=2,
    )
    axis.scatter(
        [x_values[0]],
        [proportions[0] * 100],
        s=110,
        facecolors="white",
        edgecolors="black",
        linewidth=1.1,
        zorder=3,
    )
    axis.scatter(
        [x_values[1]],
        [proportions[1] * 100],
        s=110,
        facecolors="#4d4d4d",
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )

    annotations = [
        (x_values[0], proportions[0], s_pre, n_pre, "left", 14),
        (x_values[1], proportions[1], s_post, n_post, "right", -14),
    ]
    for x_pos, prop, successes, total, ha, dx in annotations:
        axis.annotate(
            f"{_ptbr_number(prop * 100)}%\n{successes}/{total} migrações",
            xy=(x_pos, prop * 100),
            xytext=(dx, 16),
            textcoords="offset points",
            ha=ha,
            va="bottom",
        )

    axis.set_xticks(x_values, ["Pré", "Pós"])
    axis.set_xlim(-0.08, 0.70)
    axis.set_ylabel("Proporção (%)")
    axis.set_xlabel("Período")
    axis.set_title("Proporção ≤ 72 horas")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.set_ylim(0, max(ci_high * 100) * 1.32)
    _style_axis(axis)
    fig.savefig(config.figures_dir / "pre_post_72h_point_ci.png", bbox_inches="tight")
    plt.close(fig)


def plot_its_volume(df, model, x_matrix, intervention_t: int, config: AnalysisConfig) -> None:
    fitted, ci_low, ci_high = linear_predict_ci(x_matrix, model.params, model.cov_params())
    intervention_date = df.loc[df["t"] == intervention_t, "month"].iloc[0]

    fig, axis = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    axis.plot(df["month"], df["volume_total"], color="#4d4d4d", marker="o", label="Observado")
    axis.plot(df["month"], fitted, color="black", linewidth=1.9, label="Ajuste ITS")
    axis.fill_between(df["month"], ci_low, ci_high, color="black", alpha=0.10, label="IC95% do ajuste")
    axis.axvline(intervention_date, color="black", linestyle="--", linewidth=1.0, label="Intervenção")
    axis.set_title("Série temporal interrompida do volume mensal")
    axis.set_ylabel("Migrações por mês")
    axis.set_xlabel("Ano")
    _format_month_axis(axis)
    _style_axis(axis)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    fig.savefig(config.figures_dir / "its_volume_publication.png", bbox_inches="tight")
    plt.close(fig)


def plot_its_logit(df, result, x_matrix, intervention_t: int, config: AnalysisConfig) -> None:
    linear_fit, linear_low, linear_high = linear_predict_ci(x_matrix, result.params, result.cov_params())
    fitted = expit(linear_fit)
    ci_low = expit(linear_low)
    ci_high = expit(linear_high)
    observed = df["sucesso_72h"] / df["volume_total"]
    intervention_date = df.loc[df["t"] == intervention_t, "month"].iloc[0]

    fig, axis = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    axis.plot(df["month"], observed * 100, color="#4d4d4d", marker="o", label="Observado")
    axis.plot(df["month"], fitted * 100, color="black", linewidth=1.9, label="Ajuste ITS logístico")
    axis.fill_between(df["month"], ci_low * 100, ci_high * 100, color="black", alpha=0.10, label="IC95% do ajuste")
    axis.axvline(intervention_date, color="black", linestyle="--", linewidth=1.0, label="Intervenção")
    axis.set_title("Série temporal interrompida da proporção ≤ 72 horas")
    axis.set_ylabel("Proporção (%)")
    axis.set_xlabel("Ano")
    _format_month_axis(axis)
    _style_axis(axis)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    fig.savefig(config.figures_dir / "its_72h_publication.png", bbox_inches="tight")
    plt.close(fig)

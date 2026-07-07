
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


def _ptbr_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _comma_decimal_formatter(decimals: int | None = 1) -> FuncFormatter:
    return FuncFormatter(lambda value, _: _format_ptbr_tick(value, decimals))


def _format_month_axis(axis, start, end) -> None:
    axis.set_xlim(start, end)
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def _ordered_legend(axis, labels: list[str], **kwargs) -> None:
    handles, current_labels = axis.get_legend_handles_labels()
    handle_by_label = dict(zip(current_labels, handles))
    ncol = int(kwargs.get("ncol", 1) or 1)
    if ncol > 1:
        rows = (len(labels) + ncol - 1) // ncol
        labels = [
            labels[row * ncol + col]
            for col in range(ncol)
            for row in range(rows)
            if row * ncol + col < len(labels)
        ]
    axis.legend(
        [handle_by_label[label] for label in labels],
        labels,
        **kwargs,
    )


def _plot_segmented_fit_with_ci(
    axis,
    df,
    fitted,
    ci_low,
    ci_high,
    fit_label: str,
    ci_label: str,
    ci_alpha: float,
) -> None:
    for idx, period_value in enumerate((0, 1)):
        segment = df["I"] == period_value
        label_suffix = "" if idx == 0 else "_nolegend_"
        axis.fill_between(
            df.loc[segment, "month"],
            ci_low[segment],
            ci_high[segment],
            color="black",
            alpha=ci_alpha,
            label=ci_label if idx == 0 else label_suffix,
            zorder=1,
        )
        axis.plot(
            df.loc[segment, "month"],
            fitted[segment],
            color="black",
            linewidth=1.9,
            label=fit_label if idx == 0 else label_suffix,
            zorder=4,
        )


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
            f"n = {len(values)}\n$W$ = {_ptbr_number(shapiro_w, 3)}\n$p$ = {_ptbr_number(shapiro_p, 3)}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.95},
        )


def plot_qq_volume_panel(df, config: AnalysisConfig) -> None:
    pre = df.loc[df["I"] == 0, "volume_total"].to_numpy(dtype=float)
    post = df.loc[df["I"] == 1, "volume_total"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)
    _qq_panel(axes[0], pre, "Anterior")
    _qq_panel(axes[1], post, "Posterior")
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

    x_values = np.array([0.0, 0.82])
    diff_pp = (proportions[1] - proportions[0]) * 100

    fig, axis = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
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
        x_values,
        proportions * 100,
        s=70,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidth=1.1,
        zorder=3,
    )

    annotations = [
        (x_values[0], proportions[0], s_pre, n_pre),
        (x_values[1], proportions[1], s_post, n_post),
    ]
    for x_pos, prop, successes, total in annotations:
        axis.annotate(
            f"{_ptbr_number(prop * 100)}%\n{_ptbr_integer(successes)}/{_ptbr_integer(total)}",
            xy=(x_pos, prop * 100),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    axis.set_xticks(x_values, ["Anterior", "Posterior"])
    axis.set_xlim(-0.25, 1.07)
    axis.set_ylabel("Migrações concluídas em até 72 horas (%)")
    axis.set_xlabel("Período")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.set_ylim(0, max(ci_high * 100) * 1.32)
    axis.annotate(
        f"Diferença: {_ptbr_number(diff_pp)} p.p.",
        xy=(x_values.mean(), max(proportions * 100) + 3.2),
        ha="center",
        va="bottom",
    )
    _style_axis(axis)
    fig.savefig(config.figures_dir / "pre_post_72h_point_ci.png", bbox_inches="tight")
    plt.close(fig)


def plot_its_volume(df, model, x_matrix, intervention_t: int, config: AnalysisConfig) -> None:
    fitted, ci_low, ci_high = linear_predict_ci(x_matrix, model.params, model.cov_params())
    intervention_date = df.loc[df["t"] == intervention_t, "month"].iloc[0]

    fig, axis = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    _plot_segmented_fit_with_ci(
        axis,
        df,
        fitted,
        ci_low,
        ci_high,
        fit_label="Ajuste ITS linear",
        ci_label="IC95% do ajuste",
        ci_alpha=0.14,
    )
    axis.plot(df["month"], df["volume_total"], color="#4d4d4d", marker="o", markersize=4.5, label="Observado", zorder=3)
    axis.axvline(intervention_date, color="black", linestyle="--", linewidth=1.0, label="Início da implantação", zorder=5)
    axis.set_ylabel("Migrações concluídas por mês")
    axis.set_xlabel("Mês/ano")
    _format_month_axis(axis, df["month"].min(), df["month"].max())
    _style_axis(axis)
    _ordered_legend(
        axis,
        ["Observado", "Ajuste ITS linear", "IC95% do ajuste", "Início da implantação"],
        ncol=2,
        frameon=False,
        loc="upper left",
    )
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
    _plot_segmented_fit_with_ci(
        axis,
        df,
        fitted * 100,
        ci_low * 100,
        ci_high * 100,
        fit_label="Ajuste ITS logístico",
        ci_label="IC95% do ajuste",
        ci_alpha=0.12,
    )
    axis.plot(df["month"], observed * 100, color="#4d4d4d", marker="o", markersize=4.5, label="Observado", zorder=3)
    axis.axvline(intervention_date, color="black", linestyle="--", linewidth=1.0, label="Início da implantação", zorder=5)
    axis.set_ylabel("Migrações concluídas em até 72 horas (%)")
    axis.set_xlabel("Mês/ano")
    _format_month_axis(axis, df["month"].min(), df["month"].max())
    _style_axis(axis)
    _ordered_legend(
        axis,
        ["Observado", "Ajuste ITS logístico", "IC95% do ajuste", "Início da implantação"],
        ncol=2,
        frameon=False,
        loc="upper left",
    )
    fig.savefig(config.figures_dir / "its_72h_publication.png", bbox_inches="tight")
    plt.close(fig)

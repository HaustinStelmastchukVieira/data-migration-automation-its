from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import proportions_ztest


def _safe_shapiro(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    statistic, p_value = stats.shapiro(values)
    return float(statistic), float(p_value)


def _ensure_nonempty(pre: np.ndarray, post: np.ndarray) -> None:
    if len(pre) == 0 or len(post) == 0:
        raise ValueError("Amostras pré e pós não podem estar vazias.")


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    _ensure_nonempty(x, y)
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    pooled_sd = np.sqrt(((len(x) - 1) * x_var + (len(y) - 1) * y_var) / (len(x) + len(y) - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def mannwhitney_effects(u_stat: float, n_x: int, n_y: int) -> dict[str, float]:
    if n_x == 0 or n_y == 0:
        return {"r_rb_abs": float("nan"), "z_approx": float("nan"), "r_abs": float("nan")}

    u_min = min(u_stat, n_x * n_y - u_stat)
    rank_biserial = 1 - (2 * u_min) / (n_x * n_y)

    mean_u = n_x * n_y / 2
    sd_u = math.sqrt(n_x * n_y * (n_x + n_y + 1) / 12)
    z_approx = (u_stat - mean_u) / sd_u
    effect_r = abs(z_approx) / math.sqrt(n_x + n_y)

    return {
        "r_rb_abs": abs(float(rank_biserial)),
        "z_approx": float(z_approx),
        "r_abs": float(effect_r),
    }


def proportion_ci(successes: int, total: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    se = math.sqrt(proportion * (1 - proportion) / total)
    return proportion - z_value * se, proportion + z_value * se


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    mean_value = float(np.mean(values))
    if len(values) < 2:
        return mean_value, mean_value
    sem = stats.sem(values)
    low, high = stats.t.interval(1 - alpha, df=len(values) - 1, loc=mean_value, scale=sem)
    return float(low), float(high)


def pre_post_volume_tests(df: pd.DataFrame) -> dict[str, float]:
    pre = df.loc[df["I"] == 0, "volume_total"].to_numpy(dtype=float)
    post = df.loc[df["I"] == 1, "volume_total"].to_numpy(dtype=float)
    _ensure_nonempty(pre, post)

    shapiro_pre_w, shapiro_pre_p = _safe_shapiro(pre)
    shapiro_post_w, shapiro_post_p = _safe_shapiro(post)
    levene_stat, levene_p = stats.levene(pre, post, center="median")
    welch_t, welch_p = stats.ttest_ind(post, pre, equal_var=False)
    mann_whitney_u, mann_whitney_p = stats.mannwhitneyu(post, pre, alternative="two-sided")

    return {
        "n_pre": len(pre),
        "n_post": len(post),
        "mean_pre": float(np.mean(pre)),
        "mean_post": float(np.mean(post)),
        "median_pre": float(np.median(pre)),
        "median_post": float(np.median(post)),
        "mean_ci_pre_low": mean_ci(pre)[0],
        "mean_ci_pre_high": mean_ci(pre)[1],
        "mean_ci_post_low": mean_ci(post)[0],
        "mean_ci_post_high": mean_ci(post)[1],
        "shapiro_pre_W": shapiro_pre_w,
        "shapiro_pre_p": shapiro_pre_p,
        "shapiro_post_W": shapiro_post_w,
        "shapiro_post_p": shapiro_post_p,
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mw_U": float(mann_whitney_u),
        "mw_p": float(mann_whitney_p),
        "cohens_d": cohens_d(post, pre),
        **mannwhitney_effects(mann_whitney_u, len(post), len(pre)),
    }


def pre_post_sla_tests(df: pd.DataFrame) -> dict[str, float]:
    pre = df.loc[df["I"] == 0]
    post = df.loc[df["I"] == 1]

    s_pre = int(pre["sucesso_72h"].sum())
    n_pre = int(pre["volume_total"].sum())
    s_post = int(post["sucesso_72h"].sum())
    n_post = int(post["volume_total"].sum())

    if n_pre == 0 or n_post == 0:
        raise ValueError("O total pré e pós precisa ser maior que zero.")

    p_pre = s_pre / n_pre
    p_post = s_post / n_post
    diff = p_post - p_pre

    count = np.array([s_post, s_pre])
    nobs = np.array([n_post, n_pre])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    se_diff = math.sqrt(p_pre * (1 - p_pre) / n_pre + p_post * (1 - p_post) / n_post)
    diff_ci_low = diff - 1.96 * se_diff
    diff_ci_high = diff + 1.96 * se_diff

    contingency = np.array([
        [s_post, n_post - s_post],
        [s_pre, n_pre - s_pre],
    ], dtype=int)
    table = Table2x2(contingency)
    p_pre_low, p_pre_high = proportion_ci(s_pre, n_pre)
    p_post_low, p_post_high = proportion_ci(s_post, n_post)

    return {
        "S_pre": s_pre,
        "N_pre": n_pre,
        "p_pre": p_pre,
        "p_pre_ci_low": p_pre_low,
        "p_pre_ci_high": p_pre_high,
        "S_post": s_post,
        "N_post": n_post,
        "p_post": p_post,
        "p_post_ci_low": p_post_low,
        "p_post_ci_high": p_post_high,
        "diff": diff,
        "diff_ci_low": float(diff_ci_low),
        "diff_ci_high": float(diff_ci_high),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "or_est": float(table.oddsratio),
        "or_ci_low": float(table.oddsratio_confint()[0]),
        "or_ci_high": float(table.oddsratio_confint()[1]),
    }

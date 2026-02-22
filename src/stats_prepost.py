import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import Table2x2


def _safe_shapiro(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    stat, p = stats.shapiro(x)
    return float(stat), float(p)


def _ensure_nonempty(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Amostras pré e pós precisam ter pelo menos 1 observação.")


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    _ensure_nonempty(x, y)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return (np.mean(x) - np.mean(y)) / sp


def mannwhitney_effects(U_stat: float, n1: int, n2: int) -> dict:
    if n1 == 0 or n2 == 0:
        return {"r_rb_abs": float("nan"), "z_approx": float("nan"), "r_abs": float("nan")}
    U_min = min(U_stat, n1 * n2 - U_stat)
    r_rb = 1 - (2 * U_min) / (n1 * n2)

    mean_U = n1 * n2 / 2
    sd_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_approx = (U_stat - mean_U) / sd_U
    r = abs(z_approx) / math.sqrt(n1 + n2)

    return {"r_rb_abs": abs(r_rb), "z_approx": z_approx, "r_abs": r}


def pre_post_volume_tests(df: pd.DataFrame) -> dict:
    pre = df.loc[df["I"] == 0, "volume_total"].to_numpy(dtype=float)
    post = df.loc[df["I"] == 1, "volume_total"].to_numpy(dtype=float)
    _ensure_nonempty(pre, post)

    W_pre, p_pre = _safe_shapiro(pre)
    W_pos, p_pos = _safe_shapiro(post)
    lev_stat, lev_p = stats.levene(pre, post, center="median")
    t_stat, t_p = stats.ttest_ind(post, pre, equal_var=False)
    U_stat, U_p = stats.mannwhitneyu(post, pre, alternative="two-sided")
    d = cohens_d(post, pre)
    eff = mannwhitney_effects(U_stat, len(post), len(pre))

    return {
        "n_pre": len(pre),
        "n_post": len(post),
        "median_pre": float(np.median(pre)),
        "median_post": float(np.median(post)),
        "shapiro_pre_W": float(W_pre),
        "shapiro_pre_p": float(p_pre),
        "shapiro_post_W": float(W_pos),
        "shapiro_post_p": float(p_pos),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "welch_t": float(t_stat),
        "welch_p": float(t_p),
        "mw_U": float(U_stat),
        "mw_p": float(U_p),
        "cohens_d": float(d),
        **eff
    }


def pre_post_sla_tests(df: pd.DataFrame) -> dict:
    pre = df[df["I"] == 0]
    post = df[df["I"] == 1]

    S_pre = int(pre["sucesso_72h"].sum())
    N_pre = int(pre["volume_total"].sum())
    S_post = int(post["sucesso_72h"].sum())
    N_post = int(post["volume_total"].sum())
    if N_pre == 0 or N_post == 0:
        raise ValueError("Volume total não pode ser zero no pré ou pós.")

    p1 = S_pre / N_pre
    p2 = S_post / N_post
    diff = p2 - p1

    count = np.array([S_pre, S_post])
    nobs = np.array([N_pre, N_post])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    se_diff = math.sqrt(p1 * (1 - p1) / N_pre + p2 * (1 - p2) / N_post)
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    table = np.array([
        [S_post, N_post - S_post],
        [S_pre, N_pre - S_pre]
    ], dtype=int)
    tb = Table2x2(table)
    or_est = tb.oddsratio
    or_ci_low, or_ci_high = tb.oddsratio_confint()

    return {
        "S_pre": S_pre, "N_pre": N_pre, "p_pre": p1,
        "S_post": S_post, "N_post": N_post, "p_post": p2,
        "diff": diff,
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "diff_ci_low": float(ci_low),
        "diff_ci_high": float(ci_high),
        "or_est": float(or_est),
        "or_ci_low": float(or_ci_low),
        "or_ci_high": float(or_ci_high),
    }

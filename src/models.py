from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import AnalysisConfig


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(df[["t", "I", "t_post"]], has_constant="add")


def fit_its_volume_ols(df: pd.DataFrame, config: AnalysisConfig):
    x_matrix = _design_matrix(df)
    y_vector = df["volume_total"].astype(float)
    model = sm.OLS(y_vector, x_matrix).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": config.hac_lags},
    )
    return model, x_matrix


def fit_its_logit_binomial(df: pd.DataFrame, config: AnalysisConfig):
    x_matrix = _design_matrix(df)
    endog = np.column_stack([df["sucesso_72h"].to_numpy(), df["fracasso_72h"].to_numpy()])
    result = sm.GLM(endog, x_matrix, family=sm.families.Binomial()).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": config.hac_lags},
    )
    return result, x_matrix


def summarize_its_volume(model) -> pd.DataFrame:
    summary = pd.DataFrame({
        "beta": model.params,
        "IC95%_low": model.conf_int(alpha=0.05)[0],
        "IC95%_high": model.conf_int(alpha=0.05)[1],
        "p_value": model.pvalues,
    })

    post_slope_test = model.t_test([0, 1, 0, 1])
    summary.loc["post_slope_b1_plus_b3"] = {
        "beta": float(model.params["t"] + model.params["t_post"]),
        "IC95%_low": float(post_slope_test.conf_int()[0][0]),
        "IC95%_high": float(post_slope_test.conf_int()[0][1]),
        "p_value": float(post_slope_test.pvalue),
    }
    return summary


def summarize_its_logit(result) -> pd.DataFrame:
    summary = pd.DataFrame({
        "beta_logit": result.params,
        "IC95%_low": result.conf_int(alpha=0.05)[0],
        "IC95%_high": result.conf_int(alpha=0.05)[1],
        "p_value": result.pvalues,
    })
    summary["OR"] = np.exp(summary["beta_logit"])
    summary["OR_low"] = np.exp(summary["IC95%_low"])
    summary["OR_high"] = np.exp(summary["IC95%_high"])

    post_slope_test = result.t_test([0, 1, 0, 1])
    low, high = post_slope_test.conf_int()[0]
    beta = float(result.params["t"] + result.params["t_post"])
    summary.loc["post_slope_b1_plus_b3"] = {
        "beta_logit": beta,
        "IC95%_low": float(low),
        "IC95%_high": float(high),
        "p_value": float(post_slope_test.pvalue),
        "OR": float(np.exp(beta)),
        "OR_low": float(np.exp(low)),
        "OR_high": float(np.exp(high)),
    }
    return summary


def linear_predict_ci(x_matrix: pd.DataFrame, params, covariance, z_value: float = 1.959963984540054):
    design = x_matrix.to_numpy()
    beta = np.asarray(params)
    covariance_matrix = np.asarray(covariance)
    linear_prediction = design @ beta
    standard_error = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance_matrix, design))
    return linear_prediction, linear_prediction - z_value * standard_error, linear_prediction + z_value * standard_error

import numpy as np
import pandas as pd
import statsmodels.api as sm
from .config import HAC_LAGS

def fit_its_volume_ols(df: pd.DataFrame):
    X = sm.add_constant(df[["t", "I", "t_post"]], has_constant="add")
    y = df["volume_total"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return model, X

def summarize_its_volume(model) -> pd.DataFrame:
    params = model.params
    conf = model.conf_int(alpha=0.05)
    pvals = model.pvalues

    out = pd.DataFrame({
        "beta": params,
        "IC95%_low": conf[0],
        "IC95%_high": conf[1],
        "p_value": pvals
    })

    # Inclinação pós = b1 + b3
    b1 = params["t"]
    b3 = params["t_post"]
    t_test = model.t_test([0, 1, 0, 1])
    ci_post = t_test.conf_int()[0]
    p_post = float(t_test.pvalue)

    out.loc["post_slope_b1_plus_b3"] = {
        "beta": float(b1 + b3),
        "IC95%_low": float(ci_post[0]),
        "IC95%_high": float(ci_post[1]),
        "p_value": p_post
    }
    return out

def fit_its_logit_binomial(df: pd.DataFrame):
    X = sm.add_constant(df[["t", "I", "t_post"]], has_constant="add")
    endog = np.column_stack([df["sucesso_72h"].values, df["fracasso_72h"].values])

    glm = sm.GLM(endog, X, family=sm.families.Binomial())
    res = glm.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return res, X

def summarize_its_logit(res) -> pd.DataFrame:
    params = res.params
    conf = res.conf_int(alpha=0.05)
    pvals = res.pvalues

    tab = pd.DataFrame({
        "beta_logit": params,
        "IC95%_low": conf[0],
        "IC95%_high": conf[1],
        "p_value": pvals
    })
    tab["OR"] = np.exp(tab["beta_logit"])
    tab["OR_low"] = np.exp(tab["IC95%_low"])
    tab["OR_high"] = np.exp(tab["IC95%_high"])
    return tab

from src.config import AnalysisConfig
from src.io import load_monthly_data
from src.models import fit_its_logit_binomial, summarize_its_logit
from src.stats import pre_post_sla_tests


def test_pre_post_sla_z_stat_matches_positive_post_minus_pre_difference():
    config = AnalysisConfig()
    df, _ = load_monthly_data(config)
    results = pre_post_sla_tests(df)

    assert results["diff"] > 0
    assert results["z_stat"] > 0


def test_logit_summary_includes_post_slope_row():
    config = AnalysisConfig()
    df, _ = load_monthly_data(config)
    result, _ = fit_its_logit_binomial(df, config)
    summary = summarize_its_logit(result)

    assert "post_slope_b1_plus_b3" in summary.index

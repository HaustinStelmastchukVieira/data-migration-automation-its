from src.config import AnalysisConfig
from src.io import load_monthly_data


def test_load_monthly_data_shape():
    config = AnalysisConfig()
    df, intervention_t = load_monthly_data(config)

    assert len(df) == 60
    assert intervention_t == 29
    assert {"month", "volume_total", "sucesso_72h", "fracasso_72h", "t", "I", "t_post"} <= set(df.columns)

from __future__ import annotations

import json
import os

from .config import AnalysisConfig, ensure_output_dirs
from .io import load_monthly_data
from .models import (
    fit_its_logit_binomial,
    fit_its_volume_ols,
    summarize_its_logit,
    summarize_its_volume,
)
from .stats import pre_post_sla_tests, pre_post_volume_tests


def _write_json(data: dict, output_path) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def run_pipeline(config: AnalysisConfig | None = None) -> None:
    config = config or AnalysisConfig()
    ensure_output_dirs(config)

    os.environ.setdefault("MPLCONFIGDIR", str(config.mpl_config_dir))
    from .plots import (
        apply_publication_style,
        plot_its_logit,
        plot_its_volume,
        plot_pre_post_sla,
        plot_qq_volume_panel,
    )

    apply_publication_style(config)

    df, intervention_t = load_monthly_data(config)

    volume_tests = pre_post_volume_tests(df)
    sla_tests = pre_post_sla_tests(df)

    ols_model, x_ols = fit_its_volume_ols(df, config)
    ols_table = summarize_its_volume(ols_model)

    logit_model, x_logit = fit_its_logit_binomial(df, config)
    logit_table = summarize_its_logit(logit_model)

    ols_table.to_csv(config.tables_dir / "its_volume_hac6.csv")
    logit_table.to_csv(config.tables_dir / "its_logit_72h_hac6.csv")
    _write_json(volume_tests, config.tables_dir / "pre_post_volume_tests.json")
    _write_json(sla_tests, config.tables_dir / "pre_post_sla_tests.json")

    plot_qq_volume_panel(df, config)
    plot_pre_post_sla(df, config)
    plot_its_volume(df, ols_model, x_ols, intervention_t, config)
    plot_its_logit(df, logit_model, x_logit, intervention_t, config)

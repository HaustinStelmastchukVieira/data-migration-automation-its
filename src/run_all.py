import json
from pathlib import Path

from .config import TABLE_DIR
from .io_data import load_monthly_data
from .stats_prepost import pre_post_volume_tests, pre_post_sla_tests
from .its_model import (
    fit_its_volume_ols, summarize_its_volume,
    fit_its_logit_binomial, summarize_its_logit
)
from .plots import plot_qq_pre_post, plot_pre_post_sla_bars, plot_its_volume, plot_its_logit

def main() -> None:
    df, intervention_t = load_monthly_data()

    # Testes pré-pós
    vol_results = pre_post_volume_tests(df)
    sla_results = pre_post_sla_tests(df)

    # Modelos ITS
    ols_model, X_vol = fit_its_volume_ols(df)
    vol_its_table = summarize_its_volume(ols_model)

    logit_res, X_logit = fit_its_logit_binomial(df)
    logit_its_table = summarize_its_logit(logit_res)

    # Salvar tabelas
    vol_its_table.to_csv(TABLE_DIR / "its_volume_hac6.csv", index=True)
    logit_its_table.to_csv(TABLE_DIR / "its_logit_72h_hac6.csv", index=True)

    with open(TABLE_DIR / "pre_post_volume_tests.json", "w", encoding="utf-8") as f:
        json.dump(vol_results, f, ensure_ascii=False, indent=2)

    with open(TABLE_DIR / "pre_post_sla_tests.json", "w", encoding="utf-8") as f:
        json.dump(sla_results, f, ensure_ascii=False, indent=2)

    # Figuras
    plot_qq_pre_post(df)
    plot_pre_post_sla_bars(df)
    plot_its_volume(df, ols_model, X_vol, intervention_t)
    plot_its_logit(df, logit_res, X_logit, intervention_t)

    print("Concluído.")
    print(f"Figuras e tabelas salvas em: {Path(TABLE_DIR).parent}")

if __name__ == "__main__":
    main()

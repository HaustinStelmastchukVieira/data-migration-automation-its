from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AnalysisConfig


def load_monthly_data(config: AnalysisConfig, filename: str = "monthly_aggregates.csv") -> tuple[pd.DataFrame, int]:
    file_path = config.data_dir / filename
    df = pd.read_csv(file_path)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values("month").reset_index(drop=True)

    required_columns = {"month", "volume_total", "sucesso_72h"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    if len(df) != 60:
        raise ValueError(f"Esperado 60 meses, recebido {len(df)}.")
    if df["month"].isna().any():
        raise ValueError("Há datas inválidas na coluna 'month'.")
    if df["month"].duplicated().any():
        duplicated = df.loc[df["month"].duplicated(), "month"].dt.strftime("%Y-%m").tolist()
        raise ValueError(f"Há meses duplicados: {duplicated}.")
    if (df["volume_total"] < 0).any() or (df["sucesso_72h"] < 0).any():
        raise ValueError("Há valores negativos na base.")
    if (df["sucesso_72h"] > df["volume_total"]).any():
        raise ValueError("Há meses com sucesso_72h maior que volume_total.")

    intervention_ts = pd.Timestamp(f"{config.intervention_month}-01")
    if intervention_ts not in set(df["month"]):
        raise ValueError(f"Mês de intervenção {config.intervention_month} não encontrado na série.")

    df["fracasso_72h"] = df["volume_total"] - df["sucesso_72h"]
    df["t"] = range(1, len(df) + 1)
    df["I"] = (df["month"] >= intervention_ts).astype(int)

    intervention_t = int(df.loc[df["month"] == intervention_ts, "t"].iloc[0])
    df["t_post"] = ((df["t"] - intervention_t) * df["I"]).astype(int)
    return df, intervention_t

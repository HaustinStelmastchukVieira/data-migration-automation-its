import pandas as pd
from typing import Tuple
from .config import DATA_DIR, INTERVENTION_MONTH

def load_monthly_data(filename: str = "monthly_aggregates.csv") -> Tuple[pd.DataFrame, int]:
    """
    Carrega dados mensais agregados e calcula colunas auxiliares para ITS.

    Retorna:
        df: DataFrame com colunas derivadas (t, I, t_post, fracasso_72h)
        intervention_t: índice temporal (t) do mês de intervenção
    """
    df = pd.read_csv(DATA_DIR / filename)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    required = {"month", "volume_total", "sucesso_72h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    if df["month"].isna().any():
        raise ValueError("Há datas inválidas na coluna 'month'")

    if df["month"].duplicated().any():
        dupes = df.loc[df["month"].duplicated(), "month"].dt.strftime("%Y-%m").tolist()
        raise ValueError(f"Há meses duplicados em 'month': {dupes}")

    if len(df) != 60:
        raise ValueError(f"Esperado 60 meses, recebido {len(df)}")

    if (df["sucesso_72h"] > df["volume_total"]).any():
        raise ValueError("Há meses com sucesso_72h > volume_total")

    if (df["volume_total"] < 0).any() or (df["sucesso_72h"] < 0).any():
        raise ValueError("Há valores negativos nos dados")

    df["fracasso_72h"] = df["volume_total"] - df["sucesso_72h"]

    # Índice temporal 1..T
    df["t"] = range(1, len(df) + 1)

    # Intervenção (nov/2022)
    intervention_ts = pd.Timestamp(INTERVENTION_MONTH + "-01")
    if intervention_ts not in set(df["month"]):
        available = df["month"].dt.strftime("%Y-%m").tolist()
        raise ValueError(
            f"Mês de intervenção {INTERVENTION_MONTH} não encontrado. "
            f"Meses disponíveis: {available}"
        )

    df["I"] = (df["month"] >= intervention_ts).astype(int)

    # t_post: 0 no pré, 0 no mês da intervenção, 1 no mês seguinte, ...
    intervention_t = int(df.loc[df["month"] == intervention_ts, "t"].iloc[0])
    df["t_post"] = ((df["t"] - intervention_t) * df["I"]).astype(int)

    return df, intervention_t

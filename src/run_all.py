from __future__ import annotations

from .config import AnalysisConfig
from .pipeline import run_pipeline


def main() -> None:
    config = AnalysisConfig()
    run_pipeline(config)
    print("Concluído.")
    print(f"Figuras e tabelas salvas em: {config.output_dir}")


if __name__ == "__main__":
    main()

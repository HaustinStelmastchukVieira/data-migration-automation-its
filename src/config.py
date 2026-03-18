from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class AnalysisConfig:
    intervention_month: str = "2022-11"
    hac_lags: int = 6
    alpha: float = 0.05
    fig_dpi: int = 600
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def mpl_config_dir(self) -> Path:
        return self.output_dir / ".mplconfig"


def ensure_output_dirs(config: AnalysisConfig) -> None:
    for path in (config.figures_dir, config.tables_dir, config.mpl_config_dir):
        path.mkdir(parents=True, exist_ok=True)

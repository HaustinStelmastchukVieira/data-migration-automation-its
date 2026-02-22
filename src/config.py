from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MPL_CONFIG_DIR = OUTPUT_DIR / ".mplconfig"

# Analysis settings
INTERVENTION_MONTH = "2022-11"
HAC_LAGS = 6
ALPHA = 0.05
FIG_DPI = 600

# Ensure output directories exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

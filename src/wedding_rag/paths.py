from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # src/wedding_rag/paths.py → project root
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
GOLDEN_DIR = ROOT / "golden"
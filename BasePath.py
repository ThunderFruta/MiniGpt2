# Central place for mass-prediction paths used by the MiniGpt2 tools
# Set the base path (without extension) for mass prediction data

# Use the actual filename present in the workspace (capitalized)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASEPATH = BASE_DIR / "Prediction" / "MassPrediction"

INPUT_JSONL = str(BASEPATH) + ".jsonl"
OUTPUT_JSONL = str(BASEPATH) + ".outputs.jsonl"


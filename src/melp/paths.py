from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]

# TODO: configure these paths according to your environment
# Data paths - now properly using Path objects with expanduser() to handle ~
RAW_DATA_PATH = Path("~/autodl-tmp/datasets/raw").expanduser()
PROCESSED_DATA_PATH = Path("~/autodl-tmp/datasets/process").expanduser()

# Project structure paths
SPLIT_DIR = ROOT_PATH / "src/melp/data_split"
PROMPT_PATH = ROOT_PATH / "src/melp/prompt/CKEPE_prompt.json"
DATASET_LABELS_PATH = ROOT_PATH / "src/melp/prompt/dataset_class_names.json"
RESULTS_PATH = Path("~/autodl-tmp/logs/melp/results").expanduser()

# TODO: Set ECGFM pretrained model path
# Example: ECGFM_PATH = Path("~/models/ecgfm_pretrained.ckpt").expanduser()
ECGFM_PATH = Path("")  # Empty path - needs to be configured
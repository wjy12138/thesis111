import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import font_manager

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(".")
RESULTS_DIR = Path("results")

TIME_COL = "Time(year-month-day h:m:s)"
TARGET_COL = "Power (MW)"

LOOKBACK = 96 
LOOK_FORWARD = 12

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

HIDDEN_SIZE = 128
NUM_LAYERS = 4
DROPOUT = 0.4

CNN_OUT_CHANNELS = 32
CNN_KERNEL_SIZE = 3
CNN_OUT_CHANNELS_1 = 32
CNN_OUT_CHANNELS_2 = 64
CNN_KERNEL_SIZE_1 = 3
CNN_KERNEL_SIZE_2 = 3
POOL_SIZE = 2

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SELECTED_SITE_FILES = None
SELECTED_MODELS = ["LSTM", "CNN", "CNN_LSTM"]
USE_HISTORY_TARGET = True
SEQ2SEQ_TEACHER_FORCING_RATIO = 1.0

SHOW_HYPERPARAMS_IN_TITLE = True
ZOOM_PLOT_POINTS = 200
SCATTER_SAMPLE_LIMIT = 1500
FIG_DPI = 150

CHINESE_FONT_PATH = Path("fonts/NotoSansCJKsc-Regular.otf")
CHINESE_FONT_FALLBACKS = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
]


DEFAULT_CONFIG = {
    "BASE_DIR": BASE_DIR,
    "RESULTS_DIR": RESULTS_DIR,
    "TIME_COL": TIME_COL,
    "TARGET_COL": TARGET_COL,
    "LOOKBACK": LOOKBACK,
    "LOOK_FORWARD": LOOK_FORWARD,
    "TRAIN_RATIO": TRAIN_RATIO,
    "VAL_RATIO": VAL_RATIO,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LEARNING_RATE": LEARNING_RATE,
    "HIDDEN_SIZE": HIDDEN_SIZE,
    "NUM_LAYERS": NUM_LAYERS,
    "DROPOUT": DROPOUT,
    "CNN_OUT_CHANNELS": CNN_OUT_CHANNELS,
    "CNN_KERNEL_SIZE": CNN_KERNEL_SIZE,
    "CNN_OUT_CHANNELS_1": CNN_OUT_CHANNELS_1,
    "CNN_OUT_CHANNELS_2": CNN_OUT_CHANNELS_2,
    "CNN_KERNEL_SIZE_1": CNN_KERNEL_SIZE_1,
    "CNN_KERNEL_SIZE_2": CNN_KERNEL_SIZE_2,
    "POOL_SIZE": POOL_SIZE,
    "RANDOM_SEED": RANDOM_SEED,
    "DEVICE": DEVICE,
    "SELECTED_SITE_FILES": SELECTED_SITE_FILES,
    "SELECTED_MODELS": SELECTED_MODELS,
    "USE_HISTORY_TARGET": USE_HISTORY_TARGET,
    "SEQ2SEQ_TEACHER_FORCING_RATIO": SEQ2SEQ_TEACHER_FORCING_RATIO,
    "SHOW_HYPERPARAMS_IN_TITLE": SHOW_HYPERPARAMS_IN_TITLE,
    "ZOOM_PLOT_POINTS": ZOOM_PLOT_POINTS,
    "SCATTER_SAMPLE_LIMIT": SCATTER_SAMPLE_LIMIT,
    "FIG_DPI": FIG_DPI,
    "CHINESE_FONT_PATH": CHINESE_FONT_PATH,
}


def configure_environment():
    """Configure stdout/stderr to prefer UTF-8 output when possible."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def set_random_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_project_path(path_value):
    path_value = Path(path_value)
    if path_value.is_absolute():
        return path_value
    return PROJECT_DIR / path_value


def _deduplicate(items):
    unique_items = []
    seen = set()
    for item in items:
        if item and item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


def initialize_chinese_font(font_path=CHINESE_FONT_PATH):
    """Register a project font first, then fall back to common CJK system fonts."""
    font_candidates = []
    project_font_name = None

    if font_path is not None:
        resolved_font_path = _resolve_project_path(font_path)
        if resolved_font_path.exists():
            try:
                font_manager.fontManager.addfont(str(resolved_font_path))
                project_font_name = font_manager.FontProperties(fname=str(resolved_font_path)).get_name()
                font_candidates.append(project_font_name)
            except Exception as exc:
                print(
                    "Warning: failed to load project Chinese font "
                    f"{resolved_font_path}: {exc}. Falling back to system fonts."
                )

    font_candidates.extend(CHINESE_FONT_FALLBACKS)
    font_candidates = _deduplicate(font_candidates)

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = font_candidates
    matplotlib.rcParams["axes.unicode_minus"] = False

    available_names = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((name for name in font_candidates if name in available_names), None)

    if selected_font is None:
        print(
            "Warning: no usable Chinese font was found. Matplotlib will fall back to its "
            "default font and Chinese text may render incorrectly."
        )
    elif project_font_name and selected_font == project_font_name:
        print(f"Matplotlib Chinese font configured from project file: {selected_font}")
    else:
        print(f"Matplotlib Chinese font configured from system fallback: {selected_font}")

    return selected_font


def set_plot_style(font_path=CHINESE_FONT_PATH):
    """Set a plotting style that keeps Chinese labels stable across notebook environments."""
    initialize_chinese_font(font_path=font_path)
    matplotlib.rcParams["figure.dpi"] = FIG_DPI
    matplotlib.rcParams["savefig.dpi"] = FIG_DPI
    matplotlib.rcParams["axes.titlesize"] = 11
    matplotlib.rcParams["axes.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 9


def get_config(**overrides):
    """Return a copyable config dictionary with optional overrides."""
    config = deepcopy(DEFAULT_CONFIG)
    config.update(overrides)
    config["BASE_DIR"] = Path(config["BASE_DIR"])
    config["RESULTS_DIR"] = Path(config["RESULTS_DIR"])
    if config.get("CHINESE_FONT_PATH") is not None:
        config["CHINESE_FONT_PATH"] = Path(config["CHINESE_FONT_PATH"])
    return config


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(name):
    """Convert names into a filesystem-safe string."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join("_" if ch in invalid_chars else ch for ch in str(name))
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.replace("(", "").replace(")", "")
    sanitized = sanitized.replace(",", "_")
    return sanitized


def sanitize_plot_text(text):
    """Normalize plot labels into glyph-safe text for cloud notebook fonts."""
    normalized = str(text)
    replacements = {
        "\u02da": " deg",
        "\u00b0": " deg",
        "\u2218": " deg",
        "\u00b2": "^2",
        "\u00b3": "^3",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = " ".join(normalized.split())
    normalized = normalized.replace("( deg)", " (deg)")
    normalized = normalized.replace("degC", "deg C")
    return normalized


def build_hyperparam_text(config):
    return (
        f"LB={config['LOOKBACK']} | LF={config['LOOK_FORWARD']} | "
        f"BS={config['BATCH_SIZE']} | LR={config['LEARNING_RATE']} | "
        f"HS={config['HIDDEN_SIZE']} | NL={config['NUM_LAYERS']} | "
        f"DO={config['DROPOUT']} | EP={config['EPOCHS']}"
    )


def build_title_suffix(config, model_name=None):
    """Build a consistent experiment suffix for plot titles."""
    if model_name:
        title_text = model_name + "\n"
    else:
        title_text = ""

    title_text += f"LOOKBACK={config['LOOKBACK']} | LOOK_FORWARD={config['LOOK_FORWARD']}"
    if config.get("SHOW_HYPERPARAMS_IN_TITLE", True):
        title_text += "\n" + build_hyperparam_text(config)
    return title_text

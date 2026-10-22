"""Global configuration: paths, hyperparameters, temporal split, evaluation settings."""
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]      # project root
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"

FEATURES_CSV = RAW_DIR / "elliptic_txs_features.csv"
EDGELIST_CSV = RAW_DIR / "elliptic_txs_edgelist.csv"
CLASSES_CSV = RAW_DIR / "elliptic_txs_classes.csv"

# ---------- graph construction ----------
UNDIRECTED = True   # convert directed payment flows to undirected (add reverse edges, dedupe)

# ---------- leakage-free temporal split (labeled nodes only) ----------
# train: time_step <= TRAIN_TS_END
# val:   TRAIN_TS_END < time_step <= VAL_TS_END   (early stopping / threshold selection)
# test:  time_step > VAL_TS_END                   (held out, evaluated once)
TRAIN_TS_END = 29
VAL_TS_END = 34

# ---------- model hyperparameters ----------
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
GAT_HEADS = 4    # multi-head GAT; final layer is single-head to avoid dimension blow-up

# ---------- training ----------
LR = 0.01
EPOCHS = 100
WEIGHT_DECAY = 5e-4
PATIENCE = 20
LOSS = "weighted_bce"   # "weighted_bce" or "focal"
FOCAL_GAMMA = 2.0

# ---------- evaluation ----------
TOP_K = [100, 300]              # K values for Recall@K / Precision@K
SEEDS = [42, 123, 2024]         # report mean±std across seeds; use [42] for a quick run

# ---------- community detection ----------
MIN_COMMUNITY_SIZE = 50         # ignore smaller communities to reduce small-sample noise

# ---------- misc ----------
DEVICE = "cpu"                  # switch to "cuda" if a GPU is available

import json
import os

import numpy as np
import pandas as pd

SEED = 1

HIST_PATH = "ai_model_price_history.csv"   # heti ártörténet (week_start, model_id, ...)
MODELS_PATH = "ai_models_2026.csv"         # 2026-os modell-pillanatkép (képességek, limitek)
IN_COL = "input_price_per_1m_tokens"
OUT_COL = "output_price_per_1m_tokens"
PRICE_COLS = [IN_COL, OUT_COL]

WINDOW = 12          # ennyi heti múltbeli ár alapján jósolunk
HORIZON = 4          # egy jóslási lépés hossza (hét)
FFILL_LIMIT = 4      # max. ennyi hét hiányzó ár kitöltése az előző értékkel

# --- ár -> token (ársáv) ---
LOG_MIN, LOG_MAX = -3.0, 3.0     # 0,001 és 1000 USD / 1M token között
STEP = 0.05                      # egy sáv szélessége (log10 skálán) ~ 12%
N_BINS = int(round((LOG_MAX - LOG_MIN) / STEP)) + 1     # 0. token = 0 Ft-os ár, 1..120 = ársávok
VOCAB_SIZE = 2 * N_BINS          # input és output tokenek külön tartomány (az output tokenek +N_BINS)

# --- célosztályok: az ár változása sávokban, -D..+D (a nagyobb ugrás levágódik) ---
D = 20
N_CLASSES = 2 * D + 1
DELTAS = np.arange(-D, D + 1)

EMBED_DIM = 32
EPOCHS = 30
BATCH_SIZE = 256

MODEL_DIR = "models"
META_PATH = os.path.join(MODEL_DIR, "meta.json")
MODEL_KINDS = ["GlobalAvgPool", "Conv1D", "LSTM", "GRU", "BiLSTM"]


# --------------------------------------------------------------------------- #
# Adat
# --------------------------------------------------------------------------- #

### MODEL HISTORY ###
def prepare_history(df, first_week=None):
    """Tisztítás + heti sorszám (w) a teljes időtengelyen. Visszaadja: (df, first_week)."""
    df = df.dropna(subset=PRICE_COLS).copy()          # hiányzó ár nem 0, hanem ismeretlen
    df["week_start"] = pd.to_datetime(df["week_start"])
    if first_week is None:
        first_week = df["week_start"].min()
    first_week = pd.Timestamp(first_week)
    df["w"] = (df["week_start"] - first_week).dt.days // 7
    df = df.sort_values(["model_id", "w"]).reset_index(drop=True)
    return df, first_week


def price_series(g):
    """Egy modell heti ára (nyers USD), a kis lyukak kitöltve. Visszaadja: (tömb (T,2), kezdő w)."""
    g = g.set_index("w")
    full = g[PRICE_COLS].reindex(range(int(g.index.min()), int(g.index.max()) + 1))
    full = full.ffill(limit=FFILL_LIMIT)
    return full.values.astype(float), int(g.index.min())


def to_tokens(prices):
    """Ár -> ársáv-token (0 = 0 Ft; 1..N_BINS-1 = log-sávok). Bármilyen alakú tömbre működik."""
    p = np.asarray(prices, dtype=float)
    b = np.floor((np.log10(np.maximum(p, 1e-12)) - LOG_MIN) / STEP).astype(int) + 1
    b = np.clip(b, 1, N_BINS - 1)
    return np.where(p <= 0, 0, b)


def interleave(tok_window):
    """(WINDOW, 2) tokenek -> (2*WINDOW,) sorozat: in0, out0, in1, out1, ... (az output tokenek +N_BINS)."""
    return (tok_window + np.array([0, N_BINS])).reshape(-1)


def build_windows(df):
    """Minták: az utolsó WINDOW hét tokenjei -> a HORIZON héttel későbbi ár változása (osztály)."""
    X, Y, Wt, last_p, true_p = [], [], [], [], []
    for _, g in df.groupby("model_id"):
        prices, start = price_series(g)
        valid = ~np.isnan(prices).any(axis=1)
        prices = np.nan_to_num(prices)
        toks = to_tokens(prices)
        for t in range(WINDOW - 1, len(prices) - HORIZON):
            if not valid[t - WINDOW + 1:t + 1].all() or not valid[t + HORIZON]:
                continue
            X.append(interleave(toks[t - WINDOW + 1:t + 1]))
            Y.append(np.clip(toks[t + HORIZON] - toks[t], -D, D) + D)     # osztály: 0..2D
            Wt.append(start + t + HORIZON)
            last_p.append(prices[t])
            true_p.append(prices[t + HORIZON])
    return (np.array(X, dtype="int32"), np.array(Y, dtype="int32"), np.array(Wt),
            np.array(last_p), np.array(true_p))


# --------------------------------------------------------------------------- #
# Előrejelzés 
# --------------------------------------------------------------------------- #






# --------------------------------------------------------------------------- #
# Háló + tanítás
# --------------------------------------------------------------------------- #






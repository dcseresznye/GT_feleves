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


# --------------------------------------------------------------------------- #
# Adat
# --------------------------------------------------------------------------- #

### MODEL HISTORY ###
def prepare_history(df, first_week=None):     #df = dataframe itt is
    """Tisztítás + heti sorszám (w) a teljes időtengelyen. Visszaadja: (df, first_week)."""
    df = df.dropna(subset=PRICE_COLS).copy()          # hiányzó ár nem 0, hanem ismeretlen
    df["week_start"] = pd.to_datetime(df["week_start"])
    if first_week is None:
        first_week = df["week_start"].min()
    first_week = pd.Timestamp(first_week)
    df["w"] = (df["week_start"] - first_week).dt.days // 7
    df = df.sort_values(["model_id", "w"]).reset_index(drop=True)
    return df, first_week


# --------------------------------------------------------------------------- #
# Előrejelzés / app segédek
# --------------------------------------------------------------------------- #






# --------------------------------------------------------------------------- #
# Háló + tanítás
# --------------------------------------------------------------------------- #






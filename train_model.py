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

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



def decode_prices(last, p_in, p_out):
    """Osztály-valószínűségek -> várható ár. last: (n,2) mostani ár, p_*: (n, N_CLASSES).
    A várható sávváltozást alkalmazzuk az aktuális árra (a 0 Ft-os ár 0 marad)."""
    e = np.stack([np.asarray(p_in) @ DELTAS, np.asarray(p_out) @ DELTAS], axis=1)
    return np.asarray(last, dtype=float) * 10 ** (STEP * e)


def forecast(model, prices, steps):
    """Rekurzív előrejelzés: `steps` db HORIZON-hetes lépés. Visszaad (steps*HORIZON, 2) nyers árat."""
    window = np.asarray(prices[-WINDOW:], dtype=float)
    if len(window) < WINDOW or np.isnan(window).any():
        raise ValueError(f"Kevés vagy hiányos ártörténet (legalább {WINDOW} egymást követő hét kell).")
    path = []
    for _ in range(steps):
        x = interleave(to_tokens(window))[None].astype("int32")
        p_in, p_out = model.predict(x, verbose=0)                     # két softmax kimenet
        new = decode_prices(window[-1][None], np.asarray(p_in), np.asarray(p_out))[0]
        # heti értékek a mostani és a jósolt ár között (log(1+ár) skálán lineárisan)
        weekly = np.expm1(np.linspace(np.log1p(window[-1]), np.log1p(new), HORIZON + 1)[1:])
        path.append(weekly)
        window = np.vstack([window, weekly])[-WINDOW:]
    return np.vstack(path)


def active_models(df):
    """Azok a modellek, amelyek az utolsó adathéten is szerepelnek, és megvan az utolsó WINDOW hetük."""
    max_w = df["w"].max()
    recent = df[df["w"] >= max_w - WINDOW + 1]
    cnt = recent.groupby("model_id").size()
    ok = cnt[cnt == WINDOW].index
    cols = ["model_id", "provider", "model_name", "developer"]
    out = df[df["model_id"].isin(ok)].drop_duplicates("model_id", keep="last")[cols]
    return out.sort_values(["provider", "model_id"]).reset_index(drop=True)


def chart_frame(prices, start_w, path, first_week, history_weeks=52):
    """Tényleges + jósolt ár egy közös táblában, grafikonhoz."""
    first_week = pd.Timestamp(first_week)
    hist_idx = first_week + pd.to_timedelta((start_w + np.arange(len(prices))) * 7, unit="D")
    hist = pd.DataFrame(prices, index=hist_idx,
                        columns=["Input – tényleges", "Output – tényleges"]).tail(history_weeks)
    fut_idx = hist_idx[-1] + pd.to_timedelta(np.arange(1, len(path) + 1) * 7, unit="D")
    fut = pd.DataFrame(path, index=fut_idx, columns=["Input – jóslat", "Output – jóslat"])
    anchor = pd.DataFrame([hist.iloc[-1].values], index=[hist.index[-1]], columns=fut.columns)
    return pd.concat([hist, pd.concat([anchor, fut])], axis=1).sort_index()


def request_cost(in_tokens, out_tokens, requests, price_in, price_out):
    """Költség: kérések száma * (bemeneti tokenek/1M * input ár + kimeneti tokenek/1M * output ár)."""
    return requests * (in_tokens / 1e6 * price_in + out_tokens / 1e6 * price_out)







# --------------------------------------------------------------------------- #
# Háló + tanítás
# --------------------------------------------------------------------------- #

def build_model(kind):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Embedding, GlobalAveragePooling1D, Conv1D,
                                         GlobalMaxPooling1D, LSTM, GRU, Bidirectional, Dropout, Dense)

    inp = Input((2 * WINDOW,), dtype="int32", name="ar_tokenek")
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM)(inp)

    if kind == "GlobalAvgPool":
        x = GlobalAveragePooling1D()(x)
    elif kind == "Conv1D":
        x = GlobalMaxPooling1D()(Conv1D(filters=64, kernel_size=4, activation="relu")(x))
    elif kind == "LSTM":
        x = LSTM(64)(x)
    elif kind == "GRU":
        x = GRU(64)(x)
    elif kind == "BiLSTM":
        x = Bidirectional(LSTM(64))(x)
    else:
        raise ValueError(kind)

    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out_in = Dense(N_CLASSES, activation="softmax", name="input_valtozas")(x)
    out_out = Dense(N_CLASSES, activation="softmax", name="output_valtozas")(x)

    m = Model(inp, [out_in, out_out])
    m.compile(optimizer="adam", loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"])
    return m


def mae_log1p(pred, true, mask=None):
    """Átlagos abszolút hiba a log(1+ár) skálán (nagyjából relatív hiba: 0.01 ≈ 1%)."""
    err = np.abs(np.log1p(pred) - np.log1p(true))
    if mask is not None:
        if mask.sum() == 0:
            return None
        err = err[mask]
    return float(err.mean())


def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df, first_week = prepare_history(pd.read_csv(HIST_PATH))
    X, Y, Wt, last_p, true_p = build_windows(df)
    print("Minták:", X.shape, "| célok:", Y.shape)

    # Időalapú felosztás (nem véletlen): 80% múlt = tanító, 10% validáció, 10% teszt
    cut_val, cut_test = np.quantile(Wt, 0.80), np.quantile(Wt, 0.90)
    tr, va, te = Wt < cut_val, (Wt >= cut_val) & (Wt < cut_test), Wt >= cut_test
    print("train / val / test:", tr.sum(), va.sum(), te.sum())

    # a két kimenet (input ár, output ár) célosztálya: Y oszlopai
    def targets(mask):
        return [Y[mask][:, 0], Y[mask][:, 1]]

    changed = (np.abs(true_p[te] - last_p[te]) > 1e-12).any(axis=1)
    results = [{"model": "Naiv (nem változik)",
                "mae_all": mae_log1p(last_p[te], true_p[te]),
                "mae_changed": mae_log1p(last_p[te], true_p[te], changed)}]

    for kind in MODEL_KINDS:
        print(f"\n=== {kind} ===")
        net = build_model(kind)
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)   # modellenként új
        net.fit(X[tr], targets(tr), epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(X[va], targets(va)), callbacks=[es], verbose=1)
        net.save(os.path.join(MODEL_DIR, f"{kind.lower()}.keras"))
        p_in, p_out = net.predict(X[te], batch_size=1024, verbose=0)
        pred = decode_prices(last_p[te], p_in, p_out)
        results.append({"model": kind,
                        "mae_all": mae_log1p(pred, true_p[te]),
                        "mae_changed": mae_log1p(pred, true_p[te], changed)})

    nets = [r for r in results if r["model"] in MODEL_KINDS]
    best = min(nets, key=lambda r: r["mae_all"])["model"]

    meta = {
        "first_week": str(first_week.date()),
        "window": WINDOW, "horizon": HORIZON, "step": STEP, "d": D,
        "trained": MODEL_KINDS, "best": best, "results": results,
        "test_changed_share": float(changed.mean()),
        "n_samples": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nTeszt eredmények (log(1+ár) skálán mért átlagos abszolút hiba, kisebb = jobb):")
    print(pd.DataFrame(results).round(5).to_string(index=False))
    print(f"\nLegjobb háló: {best}. Mentve a '{MODEL_DIR}' mappába.")


if __name__ == "__main__":
    main()




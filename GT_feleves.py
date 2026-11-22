import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from train_model import (HIST_PATH, MODELS_PATH, IN_COL, OUT_COL, HORIZON, MODEL_DIR, META_PATH,
                         prepare_history, price_series, forecast,
                         active_models, chart_frame, request_cost)




st.set_page_config(page_title="AI modell költségkalkulátor", layout="wide")
st.title("AI modell költségkalkulátor")


# ----------------------------- adatbetöltés (gyorsítótárazva) ----------------------------- #
@st.cache_data
def load_meta():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_history(first_week):
    df, _ = prepare_history(pd.read_csv(HIST_PATH), first_week)
    return df


@st.cache_data
def load_active(first_week):
    return active_models(load_history(first_week))


@st.cache_data
def load_snapshot():
    return pd.read_csv(MODELS_PATH)


@st.cache_resource
def load_net(name):
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{name.lower()}.keras"))





# ----------------------------- előfeltételek ----------------------------- #
if not os.path.exists(META_PATH):
    st.error("Még nincs betanított modell. Futtasd előbb a terminálban: `python train_model.py`")
    st.stop()

meta = load_meta()
first_week = meta["first_week"]
df = load_history(first_week)
aktiv = load_active(first_week)
last_week = pd.Timestamp(first_week) + pd.Timedelta(weeks=int(df["w"].max()))





# ----------------------------- bemenetek ----------------------------- #
with st.sidebar:
    st.header("Beállítások")
    provider = st.selectbox("Szolgáltató", sorted(aktiv["provider"].unique()))
    model_id = st.selectbox("Modell", aktiv.loc[aktiv["provider"] == provider, "model_id"].tolist())
    hetek = st.select_slider("Hány hét múlva?", options=list(range(HORIZON, 53, HORIZON)), value=2 * HORIZON)
    net_name = st.selectbox("Előrejelző háló", meta["trained"], index=meta["trained"].index(meta["best"]))

    st.subheader("Használat")
    in_tokens = st.number_input("Bemeneti tokenek / kérés", min_value=0, value=2000, step=100)
    out_tokens = st.number_input("Kimeneti tokenek / kérés", min_value=0, value=500, step=100)
    kerelmek = st.number_input("Kérések száma", min_value=1, value=10000, step=1000)



# ----------------------------- előrejelzés ----------------------------- #
g = df[df["model_id"] == model_id]
arr, start_w = price_series(g)

with st.spinner("Előrejelzés számítása..."):
    path = forecast(load_net(net_name), arr, hetek // HORIZON)

most = arr[-1]
jovo = path[-1]
cel_datum = last_week + pd.Timedelta(weeks=hetek)

st.subheader(model_id)
st.caption(f"Az ártörténet utolsó hete: {last_week.date()}. Az előrejelzés célja: {cel_datum.date()} "
           f"({hetek} héttel az utolsó adat után). Az árak USD / 1M token (feltételezve, hogy az adatfájl így adja meg).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input ár (most)", f"{most[0]:.4f}")
c2.metric("Input ár (jóslat)", f"{jovo[0]:.4f}", f"{jovo[0] - most[0]:+.4f}")
c3.metric("Output ár (most)", f"{most[1]:.4f}")
c4.metric("Output ár (jóslat)", f"{jovo[1]:.4f}", f"{jovo[1] - most[1]:+.4f}")

k_most = request_cost(in_tokens, out_tokens, kerelmek, most[0], most[1])
k_jovo = request_cost(in_tokens, out_tokens, kerelmek, jovo[0], jovo[1])
k1, k2 = st.columns(2)
k1.metric("Költség a mostani áron (USD)", f"{k_most:,.4f}")
k2.metric("Költség a jósolt áron (USD)", f"{k_jovo:,.4f}", f"{k_jovo - k_most:+,.4f}")

st.line_chart(chart_frame(arr, start_w, path, first_week))


# ----------------------------- háttér-információk ----------------------------- #
with st.expander("Mennyire megbízható az előrejelzés?"):
    st.write("Az árak az adatban nagyon ritkán változnak, ezért a legfontosabb összevetés a "
             "**naiv módszer** („az ár nem változik”). A hiba a log(1 + ár) skálán mért átlagos "
             "abszolút eltérés, ami közelítőleg százalékos eltérésnek felel meg.")
    tabla = pd.DataFrame(meta["results"]).rename(columns={
        "model": "Modell", "mae_all": "Hiba (összes minta)", "mae_changed": "Hiba (csak árváltozásnál)"})
    tabla[["Hiba (összes minta)", "Hiba (csak árváltozásnál)"]] *= 100
    st.dataframe(tabla.round(3).fillna("–"), hide_index=True)
    st.caption(f"Értékek %-ban. A teszt minták {meta['test_changed_share']:.1%}-ában változott az ár.")
    naiv = next(r for r in meta["results"] if r["model"].startswith("Naiv"))
    sajat = next(r for r in meta["results"] if r["model"] == net_name)
    if sajat["mae_all"] >= naiv["mae_all"]:
        st.warning(f"A(z) {net_name} háló a teszten nem teljesített jobban a naiv módszernél, "
                   "ezért az előrejelzést óvatosan kezeld.")

with st.expander("A modell adatai (2026-os pillanatkép)"):
    if not os.path.exists(MODELS_PATH):
        st.caption(f"A(z) {MODELS_PATH} fájl nem található, ezért nincs kiegészítő adat.")
        st.stop()
    snap = load_snapshot()
    sor = snap[snap["model_id"] == model_id]
    if sor.empty:
        st.caption("Ehhez a modellhez nincs bejegyzés a 2026-os modellfájlban.")
    else:
        sor = sor.iloc[0]
        st.write(f"Fejlesztő országa: {sor['developer_country']} · "
                 f"max. bemenet: {sor['max_input_tokens']:,.0f} token · "
                 f"max. kimenet: {sor['max_output_tokens']:,.0f} token")
        kepessegek = [c.replace("supports_", "") for c in snap.columns
                      if c.startswith("supports_") and bool(sor[c])]
        st.write("Képességek: " + (", ".join(kepessegek) if kepessegek else "nincs jelezve"))
        if pd.notna(sor["deprecation_date"]):
            st.warning(f"Tervezett kivezetés dátuma: {sor['deprecation_date']}")


 
import pandas as pd
import numpy as np
import streamlit as st
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Flatten, Concatenate, Dense, Dropout,
                                     GRU, LSTM, Conv1D, GlobalMaxPooling1D, Cropping1D)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
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

 
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

 
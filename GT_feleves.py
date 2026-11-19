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




SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)


HIST_PATH = "ai_model_price_history.csv"   # heti ártörténet (week_start, model_id, ...)
MODELS_PATH = "ai_models_2026.csv"   # 2026-os modell-pillanatkép (képességek, limitek)
IN_COL = "input_price_per_1m_tokens"
OUT_COL = "output_price_per_1m_tokens"
 
st.set_page_config(page_title="AI modell költségkalkulátor", layout="wide")
st.title("AI modell költségkalkulátor")

pd.read_csv(HIST_PATH)

 
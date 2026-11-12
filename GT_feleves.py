import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st

st.title("CSV beolvasása")

df = pd.read_csv("ai_model_price_history.csv")  # ide írd a saját fájlod nevét

st.write("Az első 5 sor:")
st.dataframe(df.head(5))
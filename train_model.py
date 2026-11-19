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
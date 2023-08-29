import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math as math
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from joblib import parallel_backend
import plotly.graph_objects as go
from time import time
from tensorflow.keras.callbacks import TensorBoard
#from streamlit_prophet.lib.utils.load import load_config, load_image
from datetime import datetime, timedelta
import datetime 
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm


sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500 = sp500.loc["1990-01-01":].copy() 

interval_data = datetime.date(2015, 1, 1)

sp500.index = sp500.index.date

max_date = sp500.index.max()

st.info(f'Dernière date disponible :  { max_date } ',  icon="ℹ️")


st.title("Performances de nos modèles")

# Load config

# Info
with st.expander(
    "Cette page vous permettra d'évaluer nos différents models", expanded=False
):
    st.write(" Cette page permet d'afficher les performances en temps réel de nos modèles, y compris les prédictions actuelles, les erreurs, les indicateurs de performance.  La miase automatiques des données pour refléter les dernières informations du marché")
    st.write("")
st.write("")

#*************************************************


#************************ Train 

train_data = sp500[sp500.index<interval_data].Close.values

train_data_dates = sp500[sp500.index < interval_data]


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.reshape(-1, 1))

x_train = []
y_train = []

for i in range(10, len(scaled_train_data)):
    x_train.append(scaled_train_data[i - 10:i, 0])
    y_train.append(scaled_train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


#************************ Test
 

test_data = sp500[sp500.index>=interval_data].Close.values

test_data_dates = sp500[sp500.index>=interval_data].index

print(test_data_dates)

scaled_test_data = scaler.transform(test_data.reshape(-1, 1))

x_test = []
y_test = []

for i in range(10, len(scaled_test_data)):
    x_test.append(scaled_test_data[i - 10:i, 0])
    y_test.append(scaled_test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


print(y_test.shape)


@st.cache_data
def train_model(md, x_train, y_train, epochs):
    if md=="LSTM":
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    elif md=="GRU":
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(GRU(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    elif md=="CNN":
        input_shape = (x_train.shape[1], 1)
        num_classes = 2
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='relu'))


        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)
    
    return model

epochs_to_train = 50  # Par exemple, 10 époques



#






#**********************************************
 
# Calcul des erreurs résiduelles







#**********************************************

def build_graphe(md):
    value = {}

    if md == 'LSTM':
        model= train_model('LSTM', x_train, y_train, epochs_to_train)
        predictions = model.predict(x_test)
        errors_lstm = y_test - predictions.flatten()
        mean_error_lstm = np.mean(errors_lstm)
        std_error_lstm = np.std(errors_lstm)
        probabilities_lstm = norm.pdf(errors_lstm, mean_error_lstm, std_error_lstm)

        value['accuracy'] = round(r2_score(y_test, predictions), 2)
        value['mse'] = round(mean_squared_error(y_test, predictions), 2)
        value['mae'] = round(mean_absolute_error(y_test, predictions), 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data_dates[-252:], y=y_test, mode='lines', name='Données Réelles'))
        fig.add_trace(go.Scatter(x=test_data_dates[-252:], y=predictions.flatten(), mode='lines', name='Prédictions du Modèle LSTM'))
        fig.update_layout(title="Comparaison entre les Prédictions du Modèle et les Données Réelles avec le model LSTM",
                        xaxis_title="Date",
                        yaxis_title="Valeur de Clôture")
        
        plt.figure(figsize=(8, 5))
        plt.plot(errors_lstm, probabilities_lstm, 'o', alpha=0.5)
        plt.title("Graphique de Probabilité (LSTM Model)")
        plt.xlabel("Erreur Résiduelle")
        plt.ylabel("Probabilité")
    
    elif md == 'GRU':
        GRU_model= train_model('GRU', x_train, y_train, epochs_to_train)
        GRU_predictions = GRU_model.predict(x_test)
        errors_gru = y_test - GRU_predictions.flatten()
        
        mean_error_gru = np.mean(errors_gru)
        std_error_gru = np.std(errors_gru)
        probabilities_gru = norm.pdf(errors_gru, mean_error_gru, std_error_gru)

        value['accuracy'] = round(r2_score(y_test, GRU_predictions), 2)
        value['mse'] = round(mean_squared_error(y_test, GRU_predictions), 2)
        value['mae'] = round(mean_absolute_error(y_test, GRU_predictions), 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data_dates[-252:], y=y_test, mode='lines', name='Données Réelles'))
        fig.add_trace(go.Scatter(x=test_data_dates[-252:], y=GRU_predictions.flatten(), mode='lines', name='Prédictions du Modèle GRU'))
        fig.update_layout(title="Comparaison entre les Prédictions du Modèle et les Données Réelles avec le modèle GRU",
                        xaxis_title="Date",
                        yaxis_title="Valeur de Clôture")
        plt.figure(figsize=(8, 5))
        plt.plot(errors_gru, probabilities_gru, 'o', alpha=0.5)
        plt.title("Graphique de Probabilité (GRU Model)")
        plt.xlabel("Erreur Résiduelle")
        plt.ylabel("Probabilité")
    
    elif md == 'CNN':

        CNN_model= train_model('CNN', x_train, y_train, epochs_to_train)
        cnn_predicted_stock = CNN_model.predict(x_test)
        errors_cnn = y_test - cnn_predicted_stock.flatten()

        value['accuracy'] = round(r2_score(y_test, cnn_predicted_stock), 2)
        value['mse'] = round(mean_squared_error(y_test, cnn_predicted_stock), 2)
        value['mae'] = round(mean_absolute_error(y_test, cnn_predicted_stock), 2)
        
        mean_error_cnn = np.mean(errors_cnn)
        std_error_cnn = np.std(errors_cnn)
        probabilities_cnn = norm.pdf(errors_cnn, mean_error_cnn, std_error_cnn)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data_dates[-252:], y=y_test, mode='lines', name='Données Réelles'))
        fig.add_trace(go.Scatter(x=test_data_dates[-252:], y=cnn_predicted_stock.flatten(), mode='lines', name='Prédictions du Modèle CNN'))
        fig.update_layout(title="Comparaison entre les Prédictions du Modèle et les Données Réelles avec le modèle CNN",
                        xaxis_title="Date",
                        yaxis_title="Valeur de Clôture") 
        plt.figure(figsize=(8, 5))
        plt.plot(errors_cnn, probabilities_cnn, 'o', alpha=0.5)
        plt.title("Graphique de Probabilité (CNN Model)")
        plt.xlabel("Erreur Résiduelle")
        plt.ylabel("Probabilité")


    st.write("### Global performance")  

    col1, col2, col3 = st.columns(3)    
    col1.metric("Accuracy", value['accuracy'], help="Accuracy score")
    col2.metric("MSE", value['mse'], help="Mean Squared Error")
    col3.metric("MAE", value['mae'], help="Mean Absolute Error")    

    st.plotly_chart(fig)

    st.pyplot(plt)

genre = st.radio(
    "Models à évaluer : ",
    ('LSTM', 'GRU', 'CNN'))

if genre == 'LSTM':


    build_graphe('LSTM')
elif genre == 'GRU':

    build_graphe('GRU')
elif genre == 'CNN':

    build_graphe('CNN')

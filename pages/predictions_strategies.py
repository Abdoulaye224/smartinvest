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
from streamlit_lottie import st_lottie 
import json
import datetime 
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score

#********************************* Data Load ***************************************************

sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500 = sp500.loc["1990-01-01":].copy()
interval_data = datetime.date(2015, 1, 1)
sp500.index = sp500.index.date
max_date = sp500.index.max()

#*********************************** A N I M A T I O N ****************************************


def lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#chargement d'une animation 

lottie_predict = load_lottiefile("home.json")

#***************************************** ********************************************************

st.markdown("<h1 style='text-align: left; color: cadetblue;'>Analyse des Prédictions de Modèles </h1>", unsafe_allow_html=True)

st.info(f'Dernière date disponible :  { max_date } ',  icon="ℹ️")

st_lottie(lottie_predict,speed=1,reverse=False,loop=True,quality="low",height=400,width=400,key=None)



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

scaled_test_data = scaler.transform(test_data.reshape(-1, 1))

x_test = []
y_test = []

for i in range(10, len(scaled_test_data)):
    x_test.append(scaled_test_data[i - 10:i, 0])
    y_test.append(scaled_test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


print(y_test.shape)


#**************************************** Model LSTM

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=1, batch_size=32)

predictions = model.predict(x_test)
print(predictions)

#********************************************** GRU
GRU_model = Sequential()
GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
GRU_model.add(GRU(units=50, return_sequences=False))
GRU_model.add(Dense(units=1))
GRU_model.compile(optimizer='adam', loss='mean_squared_error')
GRU_model.fit(x_train, y_train, epochs=1, batch_size=32)
GRU_predictions = GRU_model.predict(x_test)

#*********************************************** CNN
input_shape = (x_train.shape[1], 1)
num_classes = 2

CNN = Sequential()
CNN.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
CNN.add(MaxPooling1D(pool_size=2))
CNN.add(LSTM(units=50, return_sequences=True))
CNN.add(Dropout(0.2))
CNN.add(Flatten())
CNN.add(Dense(1, activation='relu'))


CNN.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
CNN.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)

cnn_predicted_stock = CNN.predict(x_test)

#****************************** P R E D I C T **************************************************

num_days_to_predict = st.number_input("Nombre de jours à prédire :", min_value=1, max_value=365, value=30)  

start_date = pd.to_datetime(test_data_dates[-1]) + timedelta(days=1)
future_dates = [start_date + timedelta(days=i) for i in range(num_days_to_predict)]



x_future = x_test[-num_days_to_predict:]
predictions_future = []

for _ in range(num_days_to_predict):
    prediction = model.predict(x_future)  # Prédire le prochain jour avec LSTM
    
    predictions_future.append(prediction[0, 0]) 
    x_future = np.roll(x_future, -1)
    x_future[-1, -1] = prediction[0, 0]



future_df = pd.DataFrame({'Date': future_dates, 'Predicted': predictions_future})

st.write("Prédictions pour les {} prochains jours:".format(num_days_to_predict))
st.dataframe(future_df)

# Tracer le graphique des prédictions futures
fig = go.Figure()
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted'], mode='lines', name='Prédictions'))
fig.update_layout(title=f"Prédictions pour les {num_days_to_predict} prochains jours", xaxis_title="Date", yaxis_title="Valeur de Clôture")
st.plotly_chart(fig)


#************************** S T R A T E G Y  ******************************

def apply_strategy(long_window, short_window):
    # Vérifier que la fenêtre long terme est plus grande que la fenêtre court terme
    if long_window <= short_window:
        print("La fenêtre long terme doit être plus grande que la fenêtre court terme.")
        return
    
    df = pd.DataFrame({'Date': future_df['Date'], 'Price': future_df['Predicted']})
    df.set_index('Date', inplace=True)

    # Calculer les moyennes mobiles à court et long terme
    df['Short_MA'] = df['Price'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Price'].rolling(window=long_window).mean()

    # Initialiser les positions et les signaux
    df['Position'] = 0
    df['Signal'] = 0

    # Générer des signaux basés sur le croisement des moyennes mobiles
    for i in range(long_window, len(df)):
        if df['Short_MA'][i] > df['Long_MA'][i] and df['Short_MA'][i - 1] <= df['Long_MA'][i - 1]:
            df['Signal'][i] = 1  # Signal d'achat
        elif df['Short_MA'][i] < df['Long_MA'][i] and df['Short_MA'][i - 1] >= df['Long_MA'][i - 1]:
            df['Signal'][i] = -1  # Signal de vente

    # Appliquer les signaux aux positions
    position = 0
    for i in range(long_window, len(df)):
        if df['Signal'][i] == 1:
            position = 1
        elif df['Signal'][i] == -1:
            position = 0
        df['Position'][i] = position

    # Afficher le graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Short_MA'], mode='lines', name=f'Short MA ({short_window} days)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Long_MA'], mode='lines', name=f'Long MA ({long_window} days)'))
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Price'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Price'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))
    fig.update_layout(title='Cross Moving Average Strategy',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)
    
    # Calculer le rendement total de la stratégie en points
    total_return = 0
    position = 0
    for i in range(len(df) - 1):
        if df['Position'][i] != position:
            if df['Position'][i] == 1:  # Achat
                total_return -= df['Price'][i]
            else:  # Vente
                total_return += df['Price'][i]
            position = df['Position'][i]
    
    # Ajouter le rendement total à la liste des résultats
    results.append({'Long_Window': long_window, 'Short_Window': short_window, 'Total_Return': total_return})
#
#
#
#long_window = st.slider("Choisissez la fenêtre temporelle longue :", min_value=10, max_value=100, step=1, value=30)
#short_window = st.slider("Choisissez la fenêtre temporelle courte :", min_value=5, max_value=50, step=1, value=10)
#
#results = []
#
## Loop through different window combinations and populate the results list
#for long_window in range(10, 101):
#    for short_window in range(5, 51):
#        if long_window > short_window:
#            total_return = apply_strategy(long_window, short_window)  # Call your strategy function here
#            results.append({'Long_Window': long_window, 'Short_Window': short_window, 'Total_Return': total_return})
#
## Once all combinations have been tested, find the combination with the highest total return
#if results:
#    best_combination = max(results, key=lambda x: x['Total_Return'])
#    st.write("Résultats de chaque combinaison :")
#    for result in results:
#        st.write(f"Long Window: {result['Long_Window']}, Short Window: {result['Short_Window']}, Total Return: {result['Total_Return']:.2f} points")
#    
#    st.write("\nLa combinaison avec le meilleur rendement :")
#    st.write(f"Long Window: {best_combination['Long_Window']}, Short Window: {best_combination['Short_Window']}, Total Return: {best_combination['Total_Return']:.2f} points")
#else:
#    st.write("Aucun résultat n'a été trouvé.")
#
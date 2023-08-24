import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import requests 
from streamlit_lottie import st_lottie 

#********************* Animation
def lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#chargement d'une animation 
lottie_chargement = lottie_url("https://assets4.lottiefiles.com/packages/lf20_w51pcehl.json")
lottie_acceuil = lottie_url("https://assets8.lottiefiles.com/packages/lf20_i2eyukor.json")
lotti_inprogress = lottie_url("https://assets7.lottiefiles.com/packages/lf20_earcnwm3.json")


#*******************
sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500 = sp500.loc["1990-01-01":].copy()

data = pd.read_csv('sp500_companies.csv')

sp500.index = sp500.index.date

numeric_cols = sp500.columns

max_date = sp500.index.max()


#*****************************************

st.markdown("<h1 style='text-align: left; color: cadetblue;'>Indicateurs clés</h1>", unsafe_allow_html=True)
st.info(f'Dernière date disponible :  { max_date } ',  icon="ℹ️")

# Info
with st.expander(
    "Cette page met en évidence certains indicateurs clés utilses pour les investisseurs", expanded=False
):
    st.write(" Cette page permet d'afficher les performances en temps réel de nos modèles, y compris les prédictions actuelles, les erreurs, les indicateurs de performance.  La mise automatiques des données pour refléter les dernières informations du marché")
    
    st.write("Pour tous les indicateurs vous pourrez choisir la date de début et de fin qui vous convient. pendant la selection veuillez bien choisir une période pas seulement une date")
st.write("")


min_date = sp500.index.min()
max_date = sp500.index.max()


feature_selection = st.sidebar.multiselect(label="Choix de la valeur", options=numeric_cols)

df_features = sp500[feature_selection]

plotly_figure = px.line(data_frame=df_features, x=df_features.index, y=feature_selection, title="Chronologie")

if len(feature_selection) == 0:
    st.info('Veuillez selectionner la(es) donnée(s) à afficher dans les paramètres ! ')
else:
    st.plotly_chart(plotly_figure) 

hide_volum = st.sidebar.checkbox(label="Afficher les volume selon une période ")

if hide_volum:
    st.subheader('Le volume selon la période souhaitée')

    min_date = sp500.index.min()
    max_date = sp500.index.max()

    a_date = st.date_input("Chosir la période", (min_date, max_date), key="a1")
    
    tickerSymbol = '^GSPC'
    tickerData = yf.Ticker(tickerSymbol)

    if len(a_date) == 2:
        start_date = a_date[0].strftime("%Y-%m-%d")
        end_date = a_date[1].strftime("%Y-%m-%d")
        tickerDf = tickerData.history(period='id', start=start_date, end=end_date)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['Volume'], mode='lines', name='Volume'))
        fig.update_layout(title='Volume selon la période souhaitée', xaxis_title='Date', yaxis_title='Volume')
        
        # Display the Plotly figure using st.plotly_chart()
        st.plotly_chart(fig)
    else:
        st_lottie(lotti_inprogress,speed=1,reverse=False,loop=True,quality="low",height=200,width=200,key=None,)

a_date2 = st.date_input("Chosir la période", (min_date, max_date), key="a2")

if len(a_date2) == 2:
    filtered_df = sp500[(sp500.index >= a_date2[0]) & (sp500.index <= a_date2[1])]

    short_window = 12
    long_window = 26
    filtered_df['Short_MA'] = filtered_df['Close'].rolling(window=short_window).mean()
    filtered_df['Long_MA'] = filtered_df['Close'].rolling(window=long_window).mean()

    # Calculer le MACD Line
    filtered_df['MACD_Line'] = filtered_df['Short_MA'] - filtered_df['Long_MA']

    # Calculer la Signal Line (moyenne mobile exponentielle du MACD Line)
    signal_window = 9
    filtered_df['Signal_Line'] = filtered_df['MACD_Line'].ewm(span=signal_window, adjust=False).mean()

    # Calculer l'histogramme MACD
    filtered_df['MACD_Histogram'] = filtered_df['MACD_Line'] - filtered_df['Signal_Line']

    # Créer les traces Plotly pour chaque composant du graphique
    trace_macd_line = go.Scatter(x=filtered_df.index, y=filtered_df['MACD_Line'], mode='lines', name='MACD Line', line=dict(color='blue'))
    trace_signal_line = go.Scatter(x=filtered_df.index, y=filtered_df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='orange'))
    trace_macd_histogram = go.Bar(x=filtered_df.index, y=filtered_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='green'))

    fig = go.Figure(data=[trace_macd_line, trace_signal_line, trace_macd_histogram])
    fig.update_layout(
        title='Moving Average Convergence Divergence (MACD)',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        legend=dict(x=0.7, y=1)
    )

    st.plotly_chart(fig)

else :
    st_lottie(lotti_inprogress,speed=1,reverse=False,loop=True,quality="low",height=200,width=200,key=None,)


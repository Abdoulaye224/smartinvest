import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import requests  # pip install requests
from streamlit_lottie import st_lottie 
import json
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
lottie_chargement = lottie_url("https://assets4.lottiefiles.com/packages/lf20_w51pcehl.json")
lottie_acceuil = lottie_url("https://assets8.lottiefiles.com/packages/lf20_i2eyukor.json")
lotti_inprogress = lottie_url("https://assets7.lottiefiles.com/packages/lf20_earcnwm3.json")
lottie_predict = load_lottiefile("home.json")
lottie_acceuil = lottie_url("https://assets8.lottiefiles.com/packages/lf20_i2eyukor.json")

#********************************* M E N U ***************************************************
sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")


sp500 = sp500.loc["1990-01-01":].copy()
sp500.index = sp500.index.date

st.set_page_config(
    page_title="Accueil",
    page_icon="👋",
)


numeric_cols = sp500.columns


st.title("Bienvenue sur notre app")

# Utilisation de st.markdown() pour appliquer du style CSS
st.markdown("<h1><span style='color: cadetblue; font-size: 2.5em; font-weight: bold;'>SmartInvest</span></h1>", unsafe_allow_html=True)
st.caption(" Explorez l'avenir financier avec confiance grâce à SmartInvest - Votre plateforme interactive pour des prédictions précises, des analyses profondes et une stratégie de trading avisée.")
max_date = sp500.index.max()

st.info(f'Dernière date disponible :  { max_date } ',  icon="ℹ️")

st_lottie(lottie_acceuil,speed=1,reverse=False,loop=True,quality="low",height=None,width=None,key=None)

hide_data = st.checkbox(label="Afficher les données ")

if hide_data:
    st.dataframe(sp500[::-1])




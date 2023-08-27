import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import re
import plotly.graph_objects as go

st.title("Analyse de Sentiment")

finviz_url = 'https://finviz.com/quote.ashx?t='

# Liste des grandes entreprises du S&P 500 avec les tickers correspondants
sp500_companies = {
    'le marché SP500' : 'S&P%20500',
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'Facebook': 'META',
    # Ajoutez d'autres entreprises ici
}

# Demande à l'utilisateur de choisir le nom d'une entreprise
selected_company_name = st.selectbox("Choisissez le nom d'une entreprise du S&P 500", list(sp500_companies.keys()))

# Obtenez le ticker correspondant au nom de l'entreprise choisie
selected_ticker = sp500_companies[selected_company_name]
new_tables = {}

url = finviz_url + selected_ticker
req = Request(url=url, headers={'user-agent': 'smartinvest'})
response = urlopen(req)
html = BeautifulSoup(response, 'html')
news_table = html.find(id='news-table')
new_tables[selected_ticker] = news_table

print('selected_ticker', selected_ticker)

# ... (le reste de votre code pour collecter les données et calculer les scores de sentiment)
parsed_data = []
last_date = ""  # Initialiser la dernière date
for ticker, new_table in new_tables.items():
    for row in new_table.findAll('tr'):
        title_elem = row.find('a')  # Find the <a> tag within the row
        if title_elem is not None:  # Check if the <a> tag exists
            title = title_elem.text
        
        # Utilisation de strip() pour supprimer les espaces et sauts de ligne inutiles
        date_time_str = row.td.text.strip()
        
        date_data = date_time_str.split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
            date = last_date  # Utiliser la dernière date
        else:
            date = date_data[0]
            time = date_data[1]
            last_date = date  # Mettre à jour la dernière date
        
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']

df['compound'] = df['title'].apply(f)

df['date'] = pd.to_datetime(df.date).dt.date

mean_df = df.groupby(['date'])['compound'].mean().reset_index()


print("mean_df",mean_df.columns)
print("mean_df",mean_df['compound'])



# Créez le graphique à l'aide de Streamlit
st.title("Analyse de Sentiment")

# Graphique de Sentiment Moyen par Date
st.markdown(f"<h3>Graphique de Sentiment pour <span style='color: cadetblue;'>{selected_company_name}</span></h3>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Bar(x=mean_df.date, y=mean_df['compound'], name='score'))
st.plotly_chart(fig,  use_container_width=True)
#st.bar_chart(mean_df, x='date', y='compound')
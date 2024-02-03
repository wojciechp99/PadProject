import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
# to run this code:
# python -m streamlit run dashboard.py
st.set_page_config(layout = "wide")

initial_df = pd.read_csv('messy_data.csv', sep=', ')
df = initial_df.copy()
df.fillna(method='ffill', inplace=True)
df['table'] = pd.to_numeric(df['table'], errors='coerce')
df['table'] = pd.to_numeric(df['table'], errors='coerce')
df['table'].replace(np.nan, 58, inplace=True)
df['clarity'] = df['clarity'].str.lower()
df['color'] = df['color'].str.lower()
df['cut'] = df['cut'].str.lower()
df.drop_duplicates()
df = df.rename(columns={'x dimension': 'x', 'y dimension': 'y', 'z dimension': 'z',})

page = st.sidebar.selectbox('Select page',['First','Second']) 

if page == 'First':

    st.header("First task")
    st.write("Początkowe wartości:")
    st.dataframe(initial_df)
    st.write("Wyczyszczone wartości:")
    st.dataframe(df)
    
elif page == 'Second':

	# Wybór wartości z listył
    clist = df.select_dtypes([np.number]).columns
    value = st.selectbox("Select value:", df.columns)

    # Dopasowanie modelu i dodanie predykcji
    model = smf.ols(formula=f"price ~ {value}", data=df).fit()
    df['fitted'] = model.fittedvalues

    # Wyświetlanie wykresu
    fig = px.scatter(df, x=value, y="price", title=f"Price vs {value}",
                     labels={value: value, "price": "Price"})
    fig.add_scatter(x=df[value], y=df["fitted"], mode="lines", name="Fitted Regression Line")
    st.plotly_chart(fig)

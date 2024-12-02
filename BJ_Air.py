#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import os

from streamlit import bar_chart

st.set_page_config(layout='wide')

@st.cache_data
def load_data():
    data = pd.read_csv('~/data/Kaggle/BeijingPM.csv', index_col='No')
    data['season'] = data['season'].replace({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    year_range = data['year'].unique()
    month_range = data['month'].unique()
    season_range = data['season'].unique()

    return data, year_range, month_range, season_range

def groupby_data(data:pd.DataFrame, name:str or list, columns:list) -> pd.DataFrame:
    data = data.groupby(name)[columns].mean()
    data = data.reset_index()
    return data

df, year_range, month_range, season_range = load_data()

st.title('Beijing PM2.5')

with st.sidebar:
    month_year = st.selectbox('Select Year, Season or Month Data', ['Year', 'Month', 'Season'],
                              index=None, placeholder='Select Data')

if month_year == None:
    st.write('This is a describe statistical analysis of Beijing PM2.5 from 2010 to 2015')
    st.dataframe(df.describe().T, height=560)

if month_year == 'Year':
    year = st.slider('Select Year', year_range.min(), year_range.max())
    # with st.sidebar:
    #     year = st.slider('Select Year', year_range.min(), year_range.max())

    year_data = df[df['year'] == year]
    max_PM = year_data[['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']].agg('max')

    st.write(f"""
    #### {year} Beijing PM2.5
    """)
    m1, m2, m3, m4 = st.columns([1,1,1,1])
    m1.metric('The max PM2.5 of ' + max_PM.index[0] + 'is', int(max_PM.iloc[0]))
    m2.metric('The max PM2.5 of ' + max_PM.index[1] + 'is', int(max_PM.iloc[1]))
    m3.metric('The max PM2.5 of ' + max_PM.index[2] + 'is', int(max_PM.iloc[2]))
    m4.metric('The max PM2.5 of ' + max_PM.index[3] + 'is', int(max_PM.iloc[3]))

    col1, col2 = st.columns((1, 1))


    # plt.figure(figsize=(10, 6))
    # plt.plot(year_data.groupby('month')[['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']].mean())
    # plt.xlabel('Month')
    # plt.ylabel('PM2.5 (µg/m³)')
    # plt.legend(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'])
    #
    # col1.pyplot(plt)
    month_group = groupby_data(year_data, 'month', ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'])
    fig = px.line(month_group, x='month', y=['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'],
                 template='seaborn')
    fig.update_layout(width=600, height=400, legend=dict(orientation='h', yanchor='bottom',
                                                         y=1.02, xanchor='right', x=1))
    col1.plotly_chart(fig, use_container_width=True)

    # plt.figure(figsize=(10, 6))
    # plt.plot(year_data.groupby('month')['Iws'].sum())
    # plt.xlabel('Month')
    # plt.ylabel('Cumulative precipitation (mm)')
    #
    # col2.pyplot(plt)
    Iws_data =groupby_data(year_data, 'month',['Iws'])
    fig1 = px.line(Iws_data, x='month', y='Iws')
    fig1.update_layout(width=600, height=400)
    col2.plotly_chart(fig1, use_container_width=True)

    # plt.figure(figsize = (10, 6))
    # plt.plot(year_data.groupby('month')['TEMP'].mean())
    # plt.xlabel('Month')
    # plt.ylabel('Temperature (C)')
    #
    # col3.pyplot(plt)
    # temp_data = groupby_data(year_data, 'month', ['TEMP'])
    # fig2 = px.line(temp_data, x='month', y='TEMP')
    # fig2.update_layout(width=600, height=400)
    # col3.plotly_chart(fig2, use_container_width=True)

if month_year == 'Month':
    with st.sidebar:
        year = st.slider('Select Year', year_range.min(), year_range.max())
        month = st.slider('Select Month', month_range.min(), month_range.max())

    year_data = df[df['year'] == year]
    # month_data = year_data[year_data['month'] == month]

    st.subheader(f'{month}/{year} Beijing PM2.5')
    # plt.figure(figsize=(10, 6))
    # plt.plot(month_data.groupby('day')[['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']].mean())
    # plt.xlabel('Month')
    # plt.ylabel('PM2.5 (µg/m³)')
    # plt.legend(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'])
    #
    # st.pyplot(plt)

    day_data = groupby_data(year_data, ['month','day'],
                            ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'])
    day = day_data[day_data['month'] == month]
    fig = px.line(day, x='day', y=['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'], template='seaborn')
    fig.update_layout(width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)

if month_year == 'Season':
    with st.sidebar:
        season = st.selectbox('Select Season', season_range)
        year = st.slider('Select Year', year_range.min(), year_range.max())

    year_data = df[df['year'] == year]
    # season_data = year_data[year_data['season'] == season]
    #
    # st.subheader(f'{year} {season} Beijing PM2.5')
    # season_data1 = season_data.groupby('month')[['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']].mean()
    # x = np.arange(len(season_data1.index))
    # width = 0.2
    # multiplier = 0
    # plt.figure(figsize = (10, 6))
    # for col in season_data1.columns:
    #     offset = width * multiplier
    #     plt.bar(x + offset, season_data1[col], width)
    #     multiplier += 1
    #
    # plt.xticks(ticks=[0.3, 1.3, 2.3], labels=season_data1.index)
    # plt.legend(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'], bbox_to_anchor=(1, .25))
    # st.pyplot(plt)

    season_data = groupby_data(year_data, ['month','season'],
                               ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'])
    season_data = season_data[season_data['season'] == season]

    fig = px.bar(season_data, x='month', y=['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'],
                 barmode='group', template='seaborn')
    fig.update_xaxes(ticktext=season_data['month'], tickvals=season_data['month'])
    fig.update_layout(width=600, height=400, xaxis_title = season)
    st.plotly_chart(fig, use_container_width=True)










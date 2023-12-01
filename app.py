import scripts.pages.structured_rag as structured_rag

import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
import streamlit as st
import datetime
import plotly.express as px
import pickle as pkl

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
CACHE_DIR = BASE_DIR + os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")
MISTRAL_7B_INSTRUCT = os.getenv("MISTRAL_7B_INSTRUCT")
DB_URL = BASE_DIR + os.getenv("DB_URL")
EXPERIMENT_LOGGER = BASE_DIR + os.getenv("EXPERIMENT_LOGGER")
EXCEL_FILE = BASE_DIR + os.getenv("EXCEL_FILE")
CSV_FOLDER = BASE_DIR + os.getenv("CSV_FOLDER")
ASSET_MAPPING_PATH = BASE_DIR + os.getenv("ASSET_MAPPING_PATH")


PORTFOLIOS = [
    "low risk",
    "moderate risk",
    "medium risk",
    "high risk",
]


def create_path(path_):
    if not os.path.exists(path_.rsplit('/', 1)[0]):
        os.makedirs(path_.rsplit('/', 1)[0])
    return 

create_path(EXPERIMENT_LOGGER)
create_path(CSV_FOLDER)
create_path(DB_URL)
create_path(CACHE_DIR)

pages = [
    'Unstructured RAG',
    'Structured RAG', 
    'Custom Unstructured RAG',
    'Custom Structured RAG', 
    'History',
]

page =  st.sidebar.radio("Choose Page", pages)

st.header("FinAdvisor")


if page == 'Unstructured RAG':
        
    st.write("Coming Soon")


elif page == 'Structured RAG':
    
    st.sidebar.divider()
    portfolio = st.sidebar.selectbox("Choose Portfolio", PORTFOLIOS)
    
    chat_history_file = f"data/chat_history/chat_history_{portfolio.replace(' ', '_')}.pkl"
    but = st.sidebar.button('Clear History', use_container_width=True)
    if but:
        if os.path.exists(chat_history_file):
            os.remove(chat_history_file)
            
        
    structured_rag.render(portfolio=portfolio, history_file=chat_history_file)
    

elif page == 'General Structured RAG':
    st.write("Coming Soon")


elif page == 'General Unstructured RAG':
    
    st.write("Coming Soon")
    
elif page == 'History':
    
    try:
        df = pd.read_csv(EXPERIMENT_LOGGER)
        
        st.dataframe(df, use_container_width=True)
        
        df['response_length'] = df['llm_response'].apply(lambda x: len(x))
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            fig = px.histogram(df, x='time_taken')
            fig.update_xaxes(title='Time Taken (s)')
            fig.update_layout(title='Distribution of Time Taken to Generate Response')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            
            fig = px.histogram(df, x='response_length')
            fig.update_xaxes(title='Response Length')
            fig.update_layout(title='Distribution of Response Length')
            st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(df, x='response_length', y='time_taken')
        fig.update_xaxes(title='Response Length')
        fig.update_yaxes(title='Time Taken (s)')
        fig.update_layout(title='Time Taken with Response Length')
        st.plotly_chart(fig, use_container_width=True)
        
    except:
        st.write("No History Found")
        


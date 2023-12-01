from scripts.utils.helper_structured_rag import (
    get_database,
    get_service_context,
    get_query_engine,
    get_response,
    delete_database
)
import streamlit as st
import pandas as pd 
import pickle as pkl
import os
from dotenv import load_dotenv
import datetime
import sqlite3
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


def answer_query(query_engine, query):
    
    time, resp, sql = get_response(query_engine=query_engine, 
                                   query_str=query, 
                                   print_=False)
    return time, resp, sql


def render(portfolio=None, history_file=None):
    
    
    df = pd.read_excel(EXCEL_FILE, sheet_name=portfolio)
        
    with st.expander("Portfolio", expanded=False):
        st.dataframe(df, use_container_width=True)

    st.divider()

    with st.spinner("Initializing Database"):
        
        sql_database = get_database(DB_URL,
                                    excel_file=EXCEL_FILE,
                                    portfolio=portfolio, 
                                    csv_folder=CSV_FOLDER, 
                                    csv_path=ASSET_MAPPING_PATH)

        service_context_mistral = get_service_context(
            MISTRAL_7B_INSTRUCT, token=TOKEN, cache_dir=CACHE_DIR
        )
        
        query_engine = get_query_engine(
            sql_database=sql_database, service_context=service_context_mistral
        )

    try:
        st.session_state.messages = pkl.load(open(history_file, 'rb'))
    except:
        st.session_state.messages = []


    for message in st.session_state.messages:
        if message['role'] == 'user':
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        else:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                st.markdown(message['time_taken'])
                with st.expander('SQL', expanded=False):
                    st.code(message['sql'], language='sql')
                    if type(message['df_sql']) == str:
                        st.write(message['df_sql'])
                    else:
                        st.dataframe(message['df_sql'], hide_index=True)
                    
                    
    if input_query := st.chat_input("How can I help?"):

        st.session_state.messages.append({"role": "user", "content": input_query})

        with st.chat_message("user"):
            st.markdown(input_query)
        
        with st.spinner("Getting Response"):
            time, resp, sql = answer_query(query_engine=query_engine, query=input_query)
        
        time_taken = f"*Time Taken: {time:.3f} seconds*"
        
        try:
            conn = sqlite3.connect(DB_URL)
            df_sql = pd.read_sql(sql, con=conn)
        except:
            df_sql = 'Incorrect Query!'
        
        with st.chat_message("assistant"):
            st.markdown(resp)
            st.markdown(time_taken)
            with st.expander('SQL', expanded=True):
                st.code(sql, language='sql')
                if type(df_sql) == str:
                    st.write(df_sql)
                else:
                    st.dataframe(df_sql, hide_index=True)
            
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': resp, 
            'sql': sql, 
            'time_taken': time_taken, 
            'df_sql': df_sql
            })
        
        # log each query
        df = pd.DataFrame({
            'timestamp': [datetime.datetime.now()],
            'user_input': [input_query],
            'llm_response': [resp],
            'time_taken': [time],
            'sql_query': [sql],
            'portfolio': [portfolio], 
            })
        
        try:
            results = pd.read_csv(EXPERIMENT_LOGGER)
            results = pd.concat([results, df], axis=0, ignore_index=True)
            results.to_csv(EXPERIMENT_LOGGER, index=False)
            
        except:
            df.to_csv(EXPERIMENT_LOGGER, index=False)

    try:
        pkl.dump(st.session_state.messages, open(history_file, 'wb'))
    except:
        pass

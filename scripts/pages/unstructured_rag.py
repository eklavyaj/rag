import streamlit as st
import pandas as pd 
import pickle as pkl
import os
from dotenv import load_dotenv
import datetime
import time

from scripts.utils.helper_unstructured_rag import (
    get_service_context, 
    get_query_engine, 
    load_docs_and_save_index
)

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
CACHE_DIR = BASE_DIR + os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")
MISTRAL_7B_INSTRUCT = os.getenv("MISTRAL_7B_INSTRUCT")
DB_URL = BASE_DIR + os.getenv("DB_URL")
EXPERIMENT_LOGGER_UNSTRUCTURED = BASE_DIR + os.getenv("EXPERIMENT_LOGGER_UNSTRUCTURED")
CSV_FOLDER = BASE_DIR + os.getenv("CSV_FOLDER")
VECTOR_DB_INDEX = BASE_DIR + os.getenv("VECTOR_DB_INDEX")


def answer_query(query_engine, query):
    
    start = time.time()
    response = query_engine.query(query)
    end = time.time()
    
    return end-start, response

def clean_response(response):
    response = response.replace("$", "\$")
    return response

def render(history_file):
    
    model = st.sidebar.selectbox("Model", ['Mistral'])
    model_names = {
        'Mistral': MISTRAL_7B_INSTRUCT    
    }
    
    st.write("*Welcome, ask away!*")
    with st.spinner("Initializing App"):
        try:
            service_context = get_service_context(model_name=model_names[model], token=TOKEN, cache_dir=CACHE_DIR)
            query_engine = get_query_engine(service_context=service_context)
        except:
            
            load_docs_and_save_index(service_context=service_context)
            query_engine = get_query_engine(service_context=service_context)
            
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
                with st.expander("Sources"):
                    for i, source in enumerate(message['sources']):
                        st.link_button(source['title'], source['url'])
                        st.markdown(source['date'])
                        st.markdown(source['text'])
                        
                        if i == 3:
                            break
                    
    if input_query := st.chat_input("How can I help?"):

        st.session_state.messages.append({"role": "user", "content": input_query})

        with st.chat_message("user"):
            st.markdown(input_query)
        
        with st.spinner("Getting Response"):
            time, resp = answer_query(query_engine=query_engine, 
                                      query=input_query)
        
        source_nodes = resp.source_nodes
        sources = []
        for node in source_nodes:
            title = node.metadata['title'].replace("_", " ")
            date = title.rsplit(" ", 1)[-1].split(".")[0]
            title = title.rsplit(" ", 1)[0]

            temp = {}
            temp['url'] = ':'.join(node.text.split('\n')[0].split(":")[1:])
            temp['title'] = title
            temp['date'] = f"*Dated: {date}*"
            temp['text'] = '\n'.join(node.text.rsplit("\n", -2)[2:])
            
            sources.append(temp)
        
        resp = clean_response(resp.response)
        time_taken = f"*Time Taken: {time:.3f} seconds*"
            
        with st.chat_message("assistant"):
            st.markdown(resp)
            st.markdown(time_taken)
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.link_button(source['title'], source['url'])
                    st.markdown(source['date'])
                    st.markdown(clean_response(source['text']))
                    
                    if i == 3:
                        break
            
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': resp, 
            'time_taken': time_taken, 
            'sources': sources,
            })
        
        # log each query
        df = pd.DataFrame({
            'timestamp': [datetime.datetime.now()],
            'user_input': [input_query],
            'llm_response': [resp],
            'time_taken': [time],
            'sources': [sources]
            })
        
        try:
            results = pd.read_csv(EXPERIMENT_LOGGER_UNSTRUCTURED)
            results = pd.concat([results, df], axis=0, ignore_index=True)
            results.to_csv(EXPERIMENT_LOGGER_UNSTRUCTURED, index=False)
            
        except:
            df.to_csv(EXPERIMENT_LOGGER_UNSTRUCTURED, index=False)

    try:
        pkl.dump(st.session_state.messages, open(history_file, 'wb'))
    except:
        pass
    

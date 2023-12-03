import scripts.pages.structured_rag as structured_rag
import scripts.pages.unstructured_rag as unstructured_rag
import scripts.pages.auto_rag as auto_rag
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
import streamlit as st
import datetime
import plotly.express as px
import pickle as pkl

load_dotenv()

st.set_page_config(layout="wide")

BASE_DIR = os.getenv("BASE_DIR")
DB_URL = os.getenv("DB_URL")

CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")

MISTRAL_7B_INSTRUCT = os.getenv("MISTRAL_7B_INSTRUCT")
FINETUNED_MISTRAL = os.getenv("FINETUNED_MISTRAL")

EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_AUTO = os.getenv("EXPERIMENT_LOGGER_AUTO")
EXPERIMENT_LOGGER_STRUCTURED = os.getenv("EXPERIMENT_LOGGER_STRUCTURED")
EXPERIMENT_LOGGER_UNSTRUCTURED = os.getenv("EXPERIMENT_LOGGER_UNSTRUCTURED")

CHAT_HISTORY_AUTO = os.getenv("CHAT_HISTORY_AUTO")
CHAT_HISTORY_STRUCTURED = os.getenv("CHAT_HISTORY_STRUCTURED")
CHAT_HISTORY_UNSTRUCTURED = os.getenv("CHAT_HISTORY_UNSTRUCTURED")

VECTOR_DB_INDEX = os.getenv("VECTOR_DB_INDEX")
GRAPH_DB_INDEX = os.getenv("GRAPH_DB_INDEX")

PORTFOLIOS = [
    "low risk",
    "moderate risk",
    "medium risk",
    "high risk",
]

MODELS = [
    "Mistral-7B-Instruct",
    "Mistral-7B-Instruct-FT",
    "GPT-3.5-Turbo",
]

MODEL_NAMES = {
    "Mistral-7B-Instruct": MISTRAL_7B_INSTRUCT,
    "Mistral-7B-Instruct-FT": FINETUNED_MISTRAL,
    "GPT-3.5-Turbo": "OpenAI",
}


def create_path(path_):
    if not os.path.exists(path_.rsplit("/", 1)[0]):
        os.makedirs(path_.rsplit("/", 1)[0])
    return


create_path(EXPERIMENT_LOGGER_AUTO)
create_path(EXPERIMENT_LOGGER_STRUCTURED)
create_path(EXPERIMENT_LOGGER_UNSTRUCTURED)
create_path(DB_URL)
create_path(CACHE_DIR)
create_path(SOURCE_DOCUMENTS_PATH)
create_path(VECTOR_DB_INDEX)
create_path(GRAPH_DB_INDEX)

pages = [
    "Auto RAG",
    "Unstructured RAG",
    "Structured RAG",
    "History",
]

page = st.sidebar.radio("Choose Page", pages)

st.header("FinAdvisor")

if page == "Auto RAG":
    st.sidebar.divider()
    st.sidebar.write("Intent-based RAG over structured or unstructured Data.")

    clear_history = st.sidebar.button("Clear History", use_container_width=True)
    if clear_history:
        if os.path.exists(CHAT_HISTORY_AUTO):
            os.remove(CHAT_HISTORY_AUTO)

    auto_rag.render(
        history_file=CHAT_HISTORY_AUTO,
        models=MODELS,
        model_names=MODEL_NAMES,
        portfolios=PORTFOLIOS,
    )

elif page == "Unstructured RAG":
    st.sidebar.divider()
    st.sidebar.write(
        "RAG over news articles for multiple Stock market tickers using Vector Stores."
    )
    chat_history_file = f"data/chat_history/unstructured_rag.pkl"

    but = st.sidebar.button("Clear History", use_container_width=True)
    if but:
        if os.path.exists(chat_history_file):
            os.remove(chat_history_file)

    refresh_db = st.sidebar.button("Refresh News", use_container_width=True)
    st.sidebar.markdown("*Might take a while*")

    if refresh_db:
        service_context = unstructured_rag.get_service_context(
            model_name=MISTRAL_7B_INSTRUCT, token=TOKEN, cache_dir=CACHE_DIR
        )
        unstructured_rag.load_docs_and_save_index(service_context=service_context)

    unstructured_rag.render(history_file=chat_history_file)

elif page == "Structured RAG":
    st.sidebar.divider()
    st.sidebar.write(
        "RAG over historical stock prices for multiple Stock market tickers using Natural Language to SQL Approach."
    )
    but = st.sidebar.button("Clear History", use_container_width=True)
    if but:
        if os.path.exists(CHAT_HISTORY_STRUCTURED):
            os.remove(CHAT_HISTORY_STRUCTURED)

    structured_rag.render(
        history_file=CHAT_HISTORY_STRUCTURED,
        models=MODELS,
        model_names=MODEL_NAMES,
        portfolios=PORTFOLIOS,
    )


elif page == "History":
    rag_type = st.sidebar.selectbox(
        "RAG", ["Auto RAG", "Unstructured RAG", "Structured RAG"]
    )
    chat_history_file_dict = {"Auto RAG": ""}

    try:
        df = pd.read_csv(EXPERIMENT_LOGGER_STRUCTURED)

        st.dataframe(df, use_container_width=True)

        df["response_length"] = df["llm_response"].apply(lambda x: len(x))

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="time_taken")
            fig.update_xaxes(title="Time Taken (s)")
            fig.update_layout(title="Distribution of Time Taken to Generate Response")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x="response_length")
            fig.update_xaxes(title="Response Length")
            fig.update_layout(title="Distribution of Response Length")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df, x="response_length", y="time_taken")
        fig.update_xaxes(title="Response Length")
        fig.update_yaxes(title="Time Taken (s)")
        fig.update_layout(title="Time Taken with Response Length")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.write("No History Found")

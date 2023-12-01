import streamlit as st

import os
from dotenv import load_dotenv
import datetime
from llama_index import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    ServiceContext,
    StorageContext
)
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
import transformers
import torch
import time
from llama_index import StorageContext, load_index_from_storage

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
CACHE_DIR = BASE_DIR + os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")
MISTRAL_7B_INSTRUCT = os.getenv("MISTRAL_7B_INSTRUCT")
DB_URL = BASE_DIR + os.getenv("DB_URL")
EXPERIMENT_LOGGER_UNSTRUCTURED = BASE_DIR + os.getenv("EXPERIMENT_LOGGER_UNSTRUCTURED")
CSV_FOLDER = BASE_DIR + os.getenv("CSV_FOLDER")
VECTOR_DB_INDEX = BASE_DIR + os.getenv("VECTOR_DB_INDEX")


@st.cache_resource
def get_llm(model_name, token, cache_dir):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=token, 
        cache_dir=cache_dir
    )
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=token, 
        cache_dir=cache_dir, 
        pad_token_id=tokenizer.eos_token_id,
    )
    
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=500,
        generate_kwargs={"temperature": 0.01},
        tokenizer=tokenizer,
        model_name=model_name,
        device_map="auto",
        model_kwargs={
            "trust_remote_code": True,
            "config": model_config,
            "quantization_config": bnb_config,
            "use_auth_token": token,
            "cache_dir": cache_dir,
        },
    )
    
    return llm


@st.cache_resource
def get_service_context(model_name, token, cache_dir):
    
    llm = get_llm(model_name, token, cache_dir)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model='local:BAAI/bge-small-en')
    
    return service_context

def load_docs_and_save_index(service_context):
    
    reader = SimpleDirectoryReader(input_dir=CSV_FOLDER,
                                   required_exts=['.txt'], 
                                   recursive=True)
                                   
    docs = reader.load_data()

    for file, doc in zip(reader.input_files, docs):
        ticker = file.parts[-3]
        title = file.parts[-1]
        doc.metadata['title'] = title
        doc.metadata['ticker'] = ticker
    
    final_docs = [doc for doc in docs if 
                  "Our engineers are working quickly to resolve the issue" 
                  not in doc.text]
    
    index = VectorStoreIndex.from_documents(final_docs, service_context=service_context)
    index.storage_context.persist(VECTOR_DB_INDEX)

def get_query_engine(service_context):
    
    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_DB_INDEX)

    index = load_index_from_storage(storage_context, service_context=service_context)
    
    unstructured_query_engine = index.as_query_engine(similarity_top_k=4, response_mode="tree_summarize")
    return unstructured_query_engine
    
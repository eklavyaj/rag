from structured_rag import (
    get_database,
    get_service_context,
    get_query_engine,
    get_response,
    delete_database
)

import pandas as pd
# from llama_index import set_global_handler
# import wandb
# import llama_index
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")
MISTRAL_7B_INSTRUCT = os.getenv("MISTRAL_7B_INSTRUCT")
DB_URL = os.getenv("DB_URL")
EXPERIMENT_LOGGER = os.getenv("EXPERIMENT_LOGGER")
EXCEL_FILE = os.getenv("EXCEL_FILE")
CSV_FOLDER = os.getenv("CSV_FOLDER")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

PORTFOLIOS = [
    "low risk",
    "moderate risk",
    "medium risk",
    "high risk",
]

portfolio = PORTFOLIOS[0]

def create_path(path_):
    if not os.path.exists(path_.rsplit('/', 1)[0]):
        os.makedirs(path_.rsplit('/', 1)[0])
    return 


create_path(EXPERIMENT_LOGGER)
create_path(CSV_FOLDER)
create_path(DB_URL)
create_path(CACHE_DIR)

# delete_database(DB_URL)

# set_global_handler("wandb", run_args={"project": "llama-index"})
# wandb_callback = llama_index.global_handler

def printline(x=1):
    print("--------------------------------------------------------------------" * x)


printline()
print("Setup Log")

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


printline()


def answer_query(query_engine, query):
    
    time, resp, sql = get_response(
        query_engine=query_engine, query_str=query, print_=False
    )
    return time, resp, sql


input_queries = []
times = []
responses = []
sql_queries = []

while True:
    try:
        option = int(
            input(
"""
Menu:
    1. Enter Query
    2. Describe Database
    3. Exit

Enter Option (1 / 2 / 3): 
"""
            )
        )

    except ValueError:
        print(
"""
Invalid Option, try again.
"""
        )
        continue

    print(
f"""
Option Chosen: {option}
"""
    )

    if option == 1:
        query = input(
"""
Enter Query:

    Sample Queries:
        - What is the number of shares bought for Google in the user portfolio?
        - What is the company name for the ticker 'LULU'? 
        You may have to join with the asset_mapping table on ticker to get relevant information.
        - What asset types are contained within the user portfolio?

...
"""
        )

        input_queries.append(query)
        # print(
        #     f"""
        # User Input:
        # {query}
        # """
        # )

        print(
f"""
Getting Response...
"""
        )

        time, resp, sql = answer_query(query_engine=query_engine, query=query)

        times.append(time)
        responses.append(resp)
        sql_queries.append(sql)
        
        print(
f"""
Time Taken:
{time:.3f} seconds
"""
        )

        print(
f"""
LLM Response:
{resp}
"""
        )

        print(
f"""
SQL Query:
{sql}
"""
        )

        print("Output on Running SQL Query on Database:")
        try:
            
            conn = sqlite3.connect(DB_URL)
            c = conn.cursor()
            c.execute(sql)
            # conn.commit()
            
            tmp = c.fetchall()
            
            print(pd.DataFrame(tmp, columns=c.column_names))
            conn.close()
        except:
            print("Query Execution Failed.")
            
        printline()



    elif option == 2:
        tables = sql_database.get_usable_table_names()
        print(tables)

    else:
        try:
            df = pd.read_csv(EXPERIMENT_LOGGER)
            new_df = pd.DataFrame({
                'user_input':input_queries,
                'time_taken':times,
                'llm_response':responses,
                'sql_query':sql_queries, 
                'model_id': MISTRAL_7B_INSTRUCT,  
                'portfolio': portfolio,  
            })
            
            df = pd.concat([df, new_df], axis=0, ignore_index=True)
            df.to_csv(EXPERIMENT_LOGGER, index=False)
        
        except:
            new_df = pd.DataFrame({
                'user_input':input_queries,
                'time_taken':times,
                'llm_response':responses,
                'sql_query':sql_queries,
                'model_id': MISTRAL_7B_INSTRUCT,
                'portfolio': portfolio,  
            })
            
            new_df.to_csv(EXPERIMENT_LOGGER, index=False)
            
        print("Exiting...")
        break

# RAG

### Capstone Fall 2023

#### Python environment
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

#### If running on a GCP instance
Go to the file and add `GOOGLE_VM_CONFIG_LOCK_FILE` to `ignorable`.
```
vi .venv/lib/python3.9/site-packages/bitsandbytes/cuda_setup/env_vars.p
```
![Screenshot from 2023-12-14 05-57-38](https://github.com/eklavyaj/rag/assets/50804314/9bb1117a-9a5a-4ae4-abdf-165409b285f1)

#### Data
```
python3 ingest.py
```
#### Environment Variables
```
cp example.env .env # update necessary fields
```

#### Run streamlit app
```
streamlit run app.py
```

from google.oauth2 import service_account
from google.cloud import storage
import os
from tqdm import tqdm

credential_filename = 'Data/gcloud.json'
bucket_name = 'capstone-8e7b-4418'

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

    storage_client = storage.Client.from_service_account_json(credential_filename)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    blob_list = []
    for blob in blobs:
        blob_list.append(blob.name)
        
    return blob_list

blobs = list_blobs_with_prefix(bucket_name, 'yfinance/')

dst_blobs = [blob.replace("yfinance", "../Data/Scrapes") for blob in blobs]

def download_blob(bucket_name, blob_name, dst_path):
    """Downloads a blob into memory."""

    storage_client = storage.Client.from_service_account_json(credential_filename)#storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    dir_path = dst_path.rsplit("/", 1)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    blob.download_to_filename(dst_path)
    
for i in tqdm(range(len(blobs))):
    try:
        download_blob(bucket_name, blobs[i], dst_path=dst_blobs[i])
    except:
        continue
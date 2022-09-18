
import os
import tarfile
import boto3

def extract_model(file_name, path):
    tar = tarfile.open(f'{path}/{file_name}')
    tar.extractall(path)
    tar.close()

def get_existing_model(file_name, bucket, local_path):
    download_from_s3(bucket, f'retrain/{file_name}', f'{local_path}/{file_name}')
    extract_model(file_name, local_path)

def write_model_json(model, model_path):
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

def write_model_tar(model_dir, file_name):
    with tarfile.open(f'{model_dir}/{file_name}', "w:gz") as tar:
        for file in os.listdir(model_dir):
            tar.add(f'{model_dir}/{file}', arcname=file)

def upload_to_s3(bucket_name, path, file_name):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(f"{path}/{file_name}", bucket_name, file_name)
    except Exception as e:
        print(f'=====[S3]=====[Failed to upload {file_name}]-:', e)
        
def download_from_s3(bucket_name, key, local_file_path): # key=retrain/file_name
    s3 = boto3.client("s3")
    try:
        #download_file('mybucket', 'hello.txt', '/tmp/hello.txt')
        s3.download_file(bucket_name, key, local_file_path)
    except Exception as e:
        print(f'=====[S3]=====[Failed to download {key}]-:', e)

def remove_training_artifacts(bucket_name, file_name):
    s3 = boto3.client("s3")
    try:
        s3.delete_object(Bucket=bucket_name, Key=file_name)
    except Exception as e:
        print(f'=====[S3]=====[Failed to remove artifact {file_name}]-:', e)


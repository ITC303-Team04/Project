import json
import boto3
import uuid
import cv2 as cv

s3_client = boto3.client("s3")
S3_BUCKET_NAME = 'sagemaker-training-csu303' 
S3_DEST_BUCKET_NAME = 'preprocessed-images-csu303'
S3_PREFIX = 'sm' 

def lambda_handler(event, context): 
    # getting bucket and object key from event object
    source_bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    object_key = str(uuid.uuid4()) + '-' + key
    img_download_path = '/tmp/{}'.format(object_key)
    
    with open(img_download_path,'wb') as img_file:
        s3_client.download_fileobj(source_bucket, key, img_file)
        
        # preprocessed img path
        img_pp_path = '/tmp/pp-{}'.format(object_key)
        
        img = cv.imread(img_download_path)
        edges = cv.Canny(img,65,150)
        cv.imwrite(img_download_path, edges)
        
        # uploading img preprocessed to destination bucket
        upload_key = 'pp-{}'.format(object_key)
        s3_client.upload_file(img_download_path, S3_DEST_BUCKET_NAME,upload_key)
    
    # Delete the original file
    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
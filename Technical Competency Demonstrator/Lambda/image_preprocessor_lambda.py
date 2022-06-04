import boto3
import uuid
import cv2 as cv

s3_client = boto3.client("s3")
S3_BUCKET_NAME = "sagemaker-training-csu303"
S3_DEST_BUCKET_NAME = "preprocessed-images-csu303"
S3_PREFIX = "sm"


def get_file_from_bucket(source_bucket, key, img_file):
    s3_client.download_fileobj(source_bucket, key, img_file)


def process_image(img_path):
    img = cv.imread(img_path)
    edges = cv.Canny(img, 65, 150)
    cv.imwrite(img_path, edges)


def add_file_to_bucket(img_path, upload_key):
    s3_client.upload_file(img_path, S3_DEST_BUCKET_NAME, upload_key)


def remove_file_to_bucket(source_bucket, key):
    s3_client.delete_object(source_bucket, key)


def lambda_handler(event, context):
    # getting bucket and object key from event object
    source_bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    object_key = f"{uuid.uuid4()}-{key}"
    upload_key = f"pp-{object_key}"
    img_download_path = f"/tmp/{object_key}"

    with open(img_download_path, "wb") as img_file:
        get_file_from_bucket(source_bucket, key, img_file)
        process_image(img_download_path)
        add_file_to_bucket(img_download_path, S3_DEST_BUCKET_NAME, upload_key)

    # Delete the original file
    remove_file_to_bucket(S3_BUCKET_NAME, key)

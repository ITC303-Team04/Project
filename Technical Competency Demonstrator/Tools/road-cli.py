# A python CLI that wraps boto3 to interact with AWS  deployment
# using the API gateway.
import argparse
from unicodedata import name
import urllib.request

from parso import parse

# This is used to start the pipeline from different buckets
BUCKET_DICT = {
    "processed": "preprocessed-images-csu303",
    "mask": "mask-output-csu303"
}

API_KEY = "O7HutAb5vz1FKjQcnHfUU7fGR2jHHP8haSBd6FpO"
BUCKET = "sagemaker-training-csu303"

parser = argparse.ArgumentParser(description="CLI for AWS Sagemaker")
parser.add_argument("-p", "--position", help="Position in the pipeline. start process or mask")
parser.add_argument("-k", "--key", help="API key")
parser.add_argument("name", help="Filename to save the image as")
parser.add_argument("file", help="File path to upload")

args = parser.parse_args()

FILE_PATH = args.file

# Get the image name from command line args
IMAGE_NAME = args.name

if(args.position and args.position != "start"):
    BUCKET = BUCKET_DICT[args.position]

if(args.key):
    API_KEY = args.key

# Send a request to the API gateway to start the pipeline
def start_pipeline():
  url = f"https://dd1ixb3kb2.execute-api.ap-southeast-2.amazonaws.com/v1/{BUCKET}/{IMAGE_NAME}"
  headers = {
      "x-api-key": API_KEY,
      "Content-Type": "image/tiff"
  }

  # PUT request to start the pipeline
  response = urllib.request.Request(url, headers=headers, data=open(FILE_PATH, "rb"), method='PUT')
  response = urllib.request.urlopen(response)

  # Check if the request was successful
  if(response.getcode() == 200):
    print("Successfully started the pipeline")
  else:
    print("Error starting the pipeline")

start_pipeline()
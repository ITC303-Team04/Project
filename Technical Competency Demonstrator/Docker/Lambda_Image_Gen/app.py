import boto3
import os
import numpy as np
import uuid

import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json

def load_model(ModelPath='', WeightsPath=''):
    print("Loading model...")
    json_file = open(ModelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(WeightsPath)
    print("Model loaded.")
    return loaded_model

def make_pred_good(pred):
    print("Fixing prediction")
    pred = pred[0][:, :, :]
    pred = np.repeat(pred, 3, 2)
    print("Prediction fixed.")
    return pred

def loadImage(client, key, bucket, download_path):
    print("Loading image...")
    client.download_file(bucket, key, download_path)
    print("Image loaded.")
    return plt.imread(download_path)

def lambda_handler(event, context):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    dest_bucket_name = 'mask-output-csu303'

    s3 = boto3.client('s3')

    object_key = f'{uuid.uuid4()}-{key}'
    upload_key = f'pp-{object_key}'
    download_path = f'/tmp/{uuid.uuid4()}-{key}'
    upload_path = f'/tmp/mask_{key}'

    # Load the image
    img_shape = (512, 512)
    img = loadImage(s3, key, bucket_name, download_path) 
    img = (img/255.0)
    img = np.expand_dims(cv.resize(img, img_shape), 0)
    img = img.astype(np.float32) 

    # Handle the prediction
    print("Predicting...")
    model = load_model('model/road_seg_model.json', 'model/road_seg_weights.h5')
    pred = make_pred_good(model(img))

    # Turn np array into Tiff
    print("Turning into Image from np array")
    im = Image.fromarray((pred * 255).astype(np.uint8))
    im.save(upload_path)

    print("Uploading image")
    s3.upload_file(upload_path, dest_bucket_name, upload_key)

    print("Removing original")
    s3.delete_object(Bucket=bucket_name, Key=key)

    print("Finished.")

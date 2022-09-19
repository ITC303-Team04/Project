# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
from tensorflow import keras
import cv2 as cv

import flask
import numpy as np

model_path = "/opt/ml/model"


# The flask app for serving predictions
app = flask.Flask(__name__)

def make_good_prediction(prediction):
    prediction = prediction[0][:, :, :]
    prediction = np.repeat(prediction, 3, 2)
    return prediction

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here

    # status = 200 if health else 404
    return flask.Response(response="\n", status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    with open(os.path.join(model_path, "road_seg_model.json"), "rb") as file:
        json_model = file.read()
    model = keras.models.model_from_json(json_model)
    model.load_weights(os.path.join(model_path, "road_seg_weights.h5"))

    print(flask.request)
    data = flask.request.json
    img = np.array(data["features"])
    print(f"Image: {img}")
    img_model = model(img)
    print(f"Image Model: {img_model}")
    prediction = make_good_prediction(model(img))
    print(f"Prediction: {prediction}")

    return flask.Response(response=prediction.tobytes(), status=200)
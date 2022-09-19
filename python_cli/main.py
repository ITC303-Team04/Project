import boto3
from time import strftime, gmtime
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
from json import dumps, loads
from io import BytesIO


def create_model(sage, timestamp, role):
    model_name = "DCS-sagemaker-endpoint-" + timestamp
    primary_container = {
        "Image": "805157592068.dkr.ecr.us-east-1.amazonaws.com/itc309_testing:latest",
        "ModelDataUrl": "s3://test-sagemaker-model-jg/test_model.tar.gz",
    }
    model_response = sage.create_model(
        ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=primary_container
    )
    print(model_response["ModelArn"])

    return model_name


def create_endpoint_config(sage, timestamp, model_name):
    endpoint_config_name = "DCS-SageMaker-EndpointConfig-" + timestamp
    endpoint_config_response = sage.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.t2.large",
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

    print(f"Endpoint configuration name: {endpoint_config_name}")
    print(
        f"Endpoint configuration arn:  {endpoint_config_response['EndpointConfigArn']}"
    )
    return endpoint_config_name


def create_endpoint(sage, timestamp, endpoint_config_name):
    endpoint_name = "DCS-SageMaker-Endpoint-" + timestamp
    endpoint_params = {
        "EndpointName": endpoint_name,
        "EndpointConfigName": endpoint_config_name,
    }

    endpoint_response = sage.create_endpoint(**endpoint_params)
    print(f"EndpointArn: {endpoint_response['EndpointArn']}")

    return endpoint_name


def wait_for_endpoint(sage, endpoint_name):
    print(
        f"Waiting: {sage.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']}"
    )
    sage.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    print(
        f"Done: {sage.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']}"
    )


def main():
    timestamp = strftime("-%Y-%m-%d-%H-%M-%S", gmtime())
    sage = boto3.Session().client(service_name="sagemaker")
    runtime = boto3.Session().client(service_name="runtime.sagemaker")
    role = "arn:aws:iam::805157592068:role/service-role/AmazonSageMaker-ExecutionRole-20220801T201210"

    model_name = create_model(sage, timestamp, role)
    endpoint_config_name = create_endpoint_config(sage, timestamp, model_name)
    endpoint_name = create_endpoint(sage, timestamp, endpoint_config_name)

    img = plt.imread("./road.jpg")
    img = img / 255.0
    img = np.expand_dims(cv.resize(img, (512, 512)), 0)
    img = img.astype(np.float32)
    data = {"features": img.tolist()}
    try:
        wait_for_endpoint(sage, endpoint_name)
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name, ContentType="application/json", Body=dumps(data)
        )
    # except: 
    #     sage.delete_endpoint(EndpointName=endpoint_name)
    #     sage.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    #     sage.delete_model(ModelName=model_name)
    finally:
        sage.delete_endpoint(EndpointName=endpoint_name)
        sage.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        sage.delete_model(ModelName=model_name)
    response = response["Body"].read()
    print(response)
    res = np.frombuffer(response)
    # image = Image.open(BytesIO(response))
    image = Image.fromarray((res * 255).astype(np.uint8))
    # image = Image.fromarray((response["Body"].read() * 255).astype(np.uint8))
    image.save("output.jpg")


if __name__ == "__main__":
    main()

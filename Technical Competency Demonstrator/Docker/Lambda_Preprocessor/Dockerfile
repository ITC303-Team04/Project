# Adapted from https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-tensorflow-models-in-aws-lambda/

# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.9

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Not needed (yet) no external requirements as img processor is not finished
# Install the python requirements from requirements.txt
# RUN python3.9 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Set the CMD to your handler
CMD ["app.lambda_handler"]


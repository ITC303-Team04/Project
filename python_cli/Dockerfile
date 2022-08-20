FROM python:3.7

# RUN apt update && apt upgrade -y
RUN pip3 install pillow sagemaker boto3

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
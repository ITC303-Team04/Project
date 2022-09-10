FROM continuumio/anaconda3:latest

RUN apt update && apt upgrade -y
RUN conda install -y gdal
RUN conda install -c anaconda ipykernel matplotlib -y
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python


# docker build -t img .
# docker run --rm -it -v ${pwd}:/app img
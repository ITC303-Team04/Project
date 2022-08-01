FROM osgeo/gdal:latest

RUN apt update -y
RUN apt upgrade -y
RUN apt install python3-pip -y
RUN pip install ipykernel
COPY . /app 


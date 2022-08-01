FROM osgeo/gdal:latest

RUN apt update -y
RUN apt upgrade -y
RUN apt install python3-pip -y
RUN pip install ipykernel

# docker build -t processing .

# Windows Run cmd
# docker run -it --rm -v %cd%:/app processing 

# Unix Run cmd
# docker run -it --rm -v $(pwd):/app processing 


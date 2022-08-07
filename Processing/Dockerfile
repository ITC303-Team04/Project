# ============== CONDA/GDAL ================= 
FROM continuumio/anaconda3:latest
RUN conda install -y gdal
RUN conda install -c anaconda ipykernel -y

# docker build -t gdal .
# docker run --rm -it -v ${PWD}:/app gdal
# ==========================================



# # ============== CONDA/OPENCV =================
# FROM continuumio/anaconda3:latest
# RUN apt update && apt upgrade -y
# RUN conda install -c anaconda ipykernel -y
# RUN conda install scikit-image matplotlib -y
# RUN apt-get update && apt-get install -y python3-opencv
# RUN pip install opencv-python

# # docker build -t opencv .
# # docker run --rm -it -v ${PWD}:/app opencv
# # =============================================

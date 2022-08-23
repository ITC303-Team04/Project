FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /home/projects/docker/multi-stage-builds
COPY --from=continuumio/conda-ci-linux-64-python3.7 /opt/conda/. /opt/conda/
ENV PATH /opt/conda/bin:$PATH
RUN conda install fiona
RUN conda install rasterio
RUN conda install gdal
RUN conda install ipykernel
RUN conda install matplotlib
RUN conda install geopandas
RUN conda install pyshp
RUN conda install shapely

# docker run --gpus all -it -v $(realpath ~/projects/ITC303):/tf/ITC303 gpu-tensor-conda

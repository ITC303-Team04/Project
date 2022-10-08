FROM ubuntu:20.04

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install -y \
    wget \
    python3-pip \
    python3-setuptools \
    ca-certificates \
    libxext6 \
    libsm6 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

run pip3 --no-cache-dir install numpy tensorflow flask gunicorn pillow
run apt update && apt install -y python3-opencv

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY decision_trees /opt/program
WORKDIR /opt/program
RUN chmod +x /opt/program/serve
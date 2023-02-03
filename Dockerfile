FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04
MAINTAINER MLPerf MLBox Working Group
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            git \
            python3 \
            python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip --no-cache-dir

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt


COPY huggingartists /mlcube/huggingartists

ENV PYTHONPATH "/mlcube"

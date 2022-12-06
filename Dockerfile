ARG BASE_CONTAINER=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# ARG BASE_CONTAINER=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

FROM $BASE_CONTAINER
LABEL maintainer="Felipe Miranda <felipe.miranda@stcmed.com>"

RUN apt-get update
RUN apt-get -y install sudo
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

RUN mkdir main_dir
RUN cd main_dir
WORKDIR /main_dir
RUN mkdir env_files

ADD requirements.txt /main_dir/env_files

RUN apt update
RUN apt-get install -y python3.6

RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip3 --version
RUN pip3 install -r /main_dir/env_files/requirements.txt

RUN python3 -m ipykernel install --user --name python3

RUN python3 --version
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /main_dir/CST_v1

RUN mkdir -p /.local/share/jupyter
RUN chown -R $(id -u):$(id -g) /.local/share/jupyter
# RUN chown -R $(id -u):$(id -g) /main_dir


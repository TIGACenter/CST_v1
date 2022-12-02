ARG BASE_CONTAINER=tensorflow/tensorflow:1.12.3-gpu

FROM $BASE_CONTAINER
LABEL maintainer="Felipe Miranda <felipe.miranda@stcmed.com>"

# Added to fix pubkey problem with cuda (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN mkdir main_dir
RUN cd main_dir
WORKDIR /main_dir
RUN mkdir env_files

ADD requirements.txt /main_dir/env_files

# https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get
RUN add-apt-repository -y ppa:jblgf0/python

RUN apt update 
RUN apt-get install -y python3.6

RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN python3.6 get-pip.py

RUN python3.6 -m pip install -r /main_dir/env_files/requirements.txt

RUN python3.6 -m ipykernel install --user --name python3.6

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

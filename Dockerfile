FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="Felipe Miranda <felipe.miranda@stcmed.com>"

RUN apt-get update && apt-get install -y python3.6 wget python3-pip libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/* && mkdir -p main_dir/env_files

ADD requirements.txt /main_dir/env_files
RUN pip3 install -U pip && pip3 install -r /main_dir/env_files/requirements.txt && python3 -m ipykernel install --user --name python3

RUN useradd -ms /bin/bash docker
RUN usermod -aG 0 docker
USER docker

RUN export PATH=$PATH:~/.local/bin

# copy files into container
COPY ./ /main_dir/CST_v1/

WORKDIR /main_dir/CST_v1
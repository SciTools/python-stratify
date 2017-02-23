FROM andrewosh/binder-base

LABEL maintainer "bjlittle.pub@gmail.com"

USER main

RUN conda install -c conda-forge python-stratify matplotlib

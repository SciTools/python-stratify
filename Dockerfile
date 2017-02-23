FROM andrewosh/binder-base

LABEL maintainer "bjlittle.pub@gmail.com"

USER main
RUN conda install -c conda-forge python-stratify matplotlib=1.5.3
WORKDIR $HOME/notebooks
RUN rm -rf stratify

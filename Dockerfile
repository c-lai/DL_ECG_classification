FROM tensorflow/tensorflow:2.1.0-py3

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER clai29@jhmi.edu
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt


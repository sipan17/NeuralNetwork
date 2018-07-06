FROM tensorflow/tensorflow:1.5.1-gpu-py3
ENV ROOT=/test
RUN mkdir -p $ROOT
ADD . $ROOT
WORKDIR $ROOT
RUN pip install -r requirements.txt
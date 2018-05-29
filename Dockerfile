FROM tensorflow/tensorflow:1.5.1-py3
ENV ROOT=/nn/test
RUN mkdir -p $ROOT
ADD . $ROOT
WORKDIR $ROOT
RUN pip install -r requirements.txt
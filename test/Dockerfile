FROM python:alpine3.19

RUN apk add git
RUN apk add cargo

RUN pip install ipython

ADD ./.git  /usr/local/uut/.git
WORKDIR /usr/local/uut
RUN git checkout .

RUN pip install -e .
RUN pip install .[test]

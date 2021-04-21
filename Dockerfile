#
# keyphrase_extraction_python
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

WORKDIR app

COPY src .

# system-wide python requriments
COPY requirements.txt /tmp/requirements.txt
RUN cat /tmp/requirements.txt | xargs -n 1 -L 1 pip3 install --no-cache-dir

CMD ["bash"]
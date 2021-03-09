ARG BASE_IMAGE_VERSION=latest

FROM postgres:$BASE_IMAGE_VERSION

RUN export PG_MAJOR=`apt list --installed 2>&1 | sed -n "s/^postgresql-\([0-9.]*\)\/.*/\1/p"`             \
 && export PG_MINOR=`apt list --installed 2>&1 | sed -n "s/^postgresql-$PG_MAJOR\/\S*\s\(\S*\)\s.*/\1/p"` \
 && apt-get update \
 && apt-get install -y \
	autoconf \
    gcc \
    git \
    make \
    postgresql-server-dev-$PG_MAJOR \
	postgresql-plpython3-$PG_MAJOR \
    python3 \
    python3-pip \
	sudo \
	wget

# install all the dependencies;
# this takes a long time, so we want it to get cached,
# and it's not included in the main copy/test commands
WORKDIR /tmp/pspacy
COPY ./install_dependencies.sh /tmp/pspacy
COPY ./requirements.txt /tmp/pspacy
RUN sh install_dependencies.sh

# copy over the project and run tests
COPY . /tmp/pspacy
#RUN pip3 install flake8==3.8.4 \
 #&& flake8 --ignore=E501,E123,E402 .
#RUN python3 -m pytest
RUN make \
 && make install

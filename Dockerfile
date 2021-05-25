ARG BASE_IMAGE_VERSION=latest
FROM postgres:$BASE_IMAGE_VERSION

RUN export PG_MAJOR=`apt list --installed 2>&1 | sed -n "s/^postgresql-\([0-9.]*\)\/.*/\1/p"`             \
 && export PG_MINOR=`apt list --installed 2>&1 | sed -n "s/^postgresql-$PG_MAJOR\/\S*\s\(\S*\)\s.*/\1/p"` \
 && apt-get update \
 && apt-get install -y --no-install-recommends --allow-downgrades \
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
WORKDIR /tmp/chajda
COPY ./install_dependencies.sh /tmp/chajda
COPY ./requirements.txt /tmp/chajda

RUN sh install_dependencies.sh

# copy over the project and install
COPY . /tmp/chajda
RUN pip3 install .
RUN make \
 && make install

FROM postgres:12

RUN apt-get update && apt-get install -y \
	autoconf \
    gcc \
    git \
    make \
    postgresql-server-dev-12 \
	postgresql-plpython3-12 \
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
RUN pip3 install flake8==3.8.4 \
 && flake8 --ignore=E501,E123,E402 .
RUN python3 -m pytest \
 && make USE_PGXS=1 \
 && make USE_PGXS=1 install

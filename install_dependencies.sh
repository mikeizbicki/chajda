#!/bin/sh

set -e
working_dir=$(pwd)

pip3 install -r requirements.txt
sudachipy link -t full

# The Ukranian language needs a modified version of pymorphy2 installed
# These installation instructions are taken directly from spacy's error messages
pip3 uninstall pymorphy2 -y
pip3 install git+https://github.com/kmike/pymorphy2.git pymorphy2-dicts-uk==2.4.1.1.1460299261

# The Korean language needs the mecab-ko and mecab-ko-dic libraries,
# which unfortunately don't have convenient packaging,
# and must be installed from source.
# These installation instructions are modified from the koNLP project's instructions at:
# https://konlpy.org/en/v0.3.0/install/#optional-installations
#
# NOTE:
# The installation prefix is set to /usr instead of the default /usr/local;
# For some reason, I can't get postgres to load libraries installed in /usr/local,
# but they load just fine in /usr.
# This may interact negatively with an already installed version of mecab,
# and I have not tested this with a system that has both mecab and mecab-ko installed.

# install mecab-ko
cd $working_dir
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure --prefix=/usr
make -j
make check
sudo make install

# install mecab-ko-dic
cd $working_dir
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
autoreconf -f -i # the tar file incorrectly assumes automake-1.1, and this command reconfigures the project to use the local computer's settings; it shouldn't be necessary for properly configured projects, but in this case it seems to be
./configure --prefix=/usr
sudo ldconfig
make -j
sudo sh -c 'echo "dicdir=/usr/lib/mecab/dic/mecab-ko-dic" > /usr/etc/mecabrc'
sudo make install


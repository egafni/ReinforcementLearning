#!/bin/sh

# extract --STAGE argument value
case "$1" in
      --STAGE)
          shift
          STAGE=$1
          shift
          ;;
      *)
         echo "$1 is not a recognized flag!"
         exit 1;
         ;;
esac

apt-get update && apt-get install -y sudo htop git vim nano wget openssh-server tmux \
  python3.8 python3-pip python3.8-dev build-essential && python3.8 -m pip install --upgrade pip

ln -s /usr/bin/python3 /usr/bin/python

apt-get install -y \
  ffmpeg libsm6 libxext6 \
  freeglut3-dev \
  xvfb \
  x11-utils \
  make \
  cmake \
#  libz-de \
  libz-dev \
  libopenmpi-dev

# install TA-lib
if [ -z "$1" ]; then
  INSTALL_LOC=/usr/local
else
  INSTALL_LOC=${1}
fi
echo "Installing to ${INSTALL_LOC}"
if [ ! -f "${INSTALL_LOC}/lib/libta_lib.a" ]; then
  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install
else
  echo "TA-lib already installed, skipping installation"
fi


# xvfb/x11-utils are for creating a virtual display for openai/gym
# libopenmpi-dev is for openai/spinningup

chsh -s /bin/zsh
sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"  # good zsh theme

if [ $STAGE = "production" ] ; then
  echo "do nothing";
elif [ $STAGE = "development" ] ; then
  curl -sL https://deb.nodesource.com/setup_14.x | bash && \  # install nodejs 14
  apt-get install -y nodejs
elif [ $STAGE = "test" ] ; then
  echo "do nothing";
else exit 1
fi

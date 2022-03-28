#syntax=docker/dockerfile:1.2

FROM nvidia/cuda:11.2.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

COPY ./devops/docker_system_dependencies.sh ./devops/

# use docker build --build-arg STAGE="production"/"test" to build production/test image
ARG STAGE="development"

ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
ENV DOCKER_UID=$DOCKER_UID DOCKER_GID=$DOCKER_GID

ENV USER=ordev
ENV GROUP=ordev
ENV HOME="/home/$USER"

# installs software and creates a non-root docker user
RUN chmod +x ./devops/docker_system_dependencies.sh && \
    ./devops/docker_system_dependencies.sh --STAGE $STAGE

# create non-root docker user
RUN groupadd --gid $DOCKER_GID $GROUP && \
  useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $USER -s /bin/zsh && \
  echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# install poetry and our package
ENV POETRY_NO_INTERACTION=1 \
    # send python output directory to stdout
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# add poetry and stuff to path
ENV POETRY_HOME="$HOME/opt/poetry" \
    VENV_PATH="$HOME/.local/"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH" POETRY_VERSION=1.1.4

RUN mkdir $HOME/opt/ && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.8 - &&\
    poetry config virtualenvs.create false &&\
    mkdir -p /static/.cache/poetry &&\
    poetry config cache-dir /static/.cache/poetry

WORKDIR $HOME/TheOracle

# install core (production) dependencies - poetry dependency caching
COPY pyproject.toml poetry.lock ./
COPY submodules ./submodules

RUN sed -i 's/#\!\/usr\/bin\/env python3/#\!\/usr\/bin\/env python3.8/g' $POETRY_HOME/bin/poetry  # dirty hack to use correct python for poetry

RUN --mount=type=cache,target=/static/.cache/poetry \
    poetry install --no-dev --no-root  # no-root for more efficient docker caching

# optionally install development/test depencendencies
RUN --mount=type=cache,target=/static/.cache/poetry \
    if [ "$STAGE" = "production" ] ; then poetry install --no-dev --no-root; \
    elif [ "$STAGE" = "development" ] ; then poetry install --no-root; \
    elif [ "$STAGE" = "test" ] ; then poetry install --no-root; \
    else false ; fi

# install project - run this the last for project code caching
COPY rl/ ./rl/
RUN --mount=type=cache,target=/static/.cache/poetry \
    if [ "$STAGE" = "production" ] ; then poetry install --no-dev --no-root; \
    elif [ "$STAGE" = "development" ] ; then poetry install --no-root; \
    elif [ "$STAGE" = "test" ] ; then poetry install --no-root; \
    else false ; fi

RUN rm -rf /static/.cache/*

## temporary thing to fix torch with 30xx gpus

RUN python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

RUN sudo chown -R $USER:$GROUP $HOME

# Jupyter settings
RUN mkdir -p $HOME/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/shortcuts.jupyterlab-settings
COPY ./devops/shortcuts.jupyter-lab-settings $HOME/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/shortcuts.jupyterlab-settings

RUN env
RUN ls -al
RUN which python

CMD [ "" ]

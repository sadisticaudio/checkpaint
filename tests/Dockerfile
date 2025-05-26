FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

# Set up the base Conda environment by installing PyTorch and friends.
COPY conda-linux-64.lock /app/conda-linux-64.lock
RUN micromamba create -qy -n base -f /app/conda-linux-64.lock \
 && rm /app/conda-linux-64.lock \
 && micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX" \
 && micromamba clean -qya

RUN conda init && bash ~/.bashrc && . ~/.bashrc

# Fix for https://github.com/pytorch/pytorch/issues/97041
RUN ln -s "$CONDA_PREFIX/lib/libnvrtc.so.11.8.89" "$CONDA_PREFIX/lib/libnvrtc.so"

USER user
WORKDIR /app

COPY configureDockerEnvs.sh configureDockerEnvs.sh
RUN ./configureDockerEnvs.sh
COPY configureDockerEnvs2.4.sh configureDockerEnvs2.4.sh
RUN ./configureDockerEnvs2.4.sh
COPY configureDockerEnvs2.4fix.sh configureDockerEnvs2.4fix.sh
RUN ./configureDockerEnvs2.4fix.sh
COPY dist/*.whl .
COPY grok grokking
COPY dockercimporter.py cimporter.py
COPY testWheels.sh testWheels.sh
RUN ./testWheels.sh

# Set the default command to python3.
CMD ["/bin/bash"]

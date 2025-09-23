FROM ubuntu:latest

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# APT Package Manager + LibTorch
COPY Dockerfile.packages packages
RUN apt-get update -qq \
    && ln -sf /usr/share/zoneinfo/UTC /etc/localtime \
    && apt-get -y install $(awk -F '#' '{print $1}' packages) \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/cache/apt/archives/* && rm -rf /var/lib/apt/lists/*

# Conda Environment Manager
ENV PATH=/opt/conda/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN SHFILE="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && wget -O conda.sh $SHFILE && sh ./conda.sh -b -p "/opt/conda" && rm conda.sh

WORKDIR /root
COPY .condarc .
COPY .bashrc .
COPY .env .

CMD ["bash"]

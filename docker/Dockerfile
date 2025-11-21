FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

LABEL maintainer="Rustam Akhmadiev" \
      email="akhmarust@sep.stanford.edu"

# Set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install essential packages
RUN apt-get -y update && \
    apt-get install -y tzdata && \
    apt-get -y install g++ python3.10 python3.10-dev python3.10-venv python3-pip python3-numpy \
                    git make gcc libboost-all-dev libboost-dev libtbb-dev \
                    gfortran libfftw3-3 libfftw3-dev libssl-dev \
                    flex libxaw7-dev vim wget software-properties-common \
                    ca-certificates lsb-release && \
    apt-get -y clean

# Make Python 3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Update pip and set Python symlink
RUN python3 -m pip install --upgrade pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install cmake with Python 3.10
RUN python3 -m pip install cmake --upgrade
ENV PATH=~/.local/bin:$PATH

# Install pybind11
RUN git clone https://github.com/pybind/pybind11.git /opt/pybind11/src && \
    cd /opt/pybind11/src && git checkout 75007dda72ad4508064c1f080394eae50d3a61ee && \
    mkdir -p /opt/pybind11/build && \
    cd /opt/pybind11/build && \
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_EXECUTABLE=$(which python3) ../src && \
    make -j 4 install

# Build OpenCV
RUN git clone https://github.com/opencv/opencv.git /opt/opencv/src && \
    cd /opt/opencv/src && git checkout 4.x && \
    mkdir -p /opt/opencv/build && cd /opt/opencv/build && \
    cmake -DBUILD_LIST=core -DPYTHON_EXECUTABLE=$(which python3) ../src && \
    make -j8 install

# Build SEP IO libraries
ENV sepflags="-O3 -ffast-math -DNDEBUG -DBOOST_DISABLE_ASSERTS -funroll-loops"
RUN git clone http://zapad.stanford.edu/SEP-external/sep-iolibs.git /opt/sep-io/src && \
    mkdir -p /opt/sep-io/build && \
    cd /opt/sep-io/src && git submodule update --init --recursive && \
    cd external/genericIO && git checkout rustam-dev && \
    cd /opt/sep-io/src && git add . && \
    cd /opt/sep-io/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/SEP -DCMAKE_CXX_FLAGS="$sepflags" \
    -DBUILD_TEST=ON -DBUILD_SEP=ON -DBUILD_SEGYIO=OFF -DBUILD_GCP=OFF -DBUILD_UTIL=OFF \
    -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    ../src && \
    make -j8 install

# Clone python-solver
RUN git clone --branch rustam-dev http://cees-gitlab.stanford.edu/ettore88/python-solver.git /opt/python-solver

# Install Python packages
RUN python3 -m pip install notebook h5py numba scipy matplotlib

# Install additional Python packages
RUN python3 -m pip install dask distributed dask_jobqueue && \
    python3 -m pip install "holoviews[recommended]" && \
    python3 -m pip install numba &&\
    pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre jupyterlab_nvdashboard

# Install Arrow
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get update && \
    apt-get install -y -V libarrow-dev && \
    rm -f ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    python3 -m pip install pyarrow

# Install compression
RUN git clone https://github.com/szcompressor/SZ3.git /opt/sz3/src && \
    mkdir -p /opt/sz3/build && cd /opt/sz3/build &&\
    cmake ../src && make -j8 install

# Create and set working directory
RUN mkdir -p /home
WORKDIR /home

# Set environment variables
ENV TZ=America/Los_Angeles
ENV PATH=~/.local/bin:$PATH
ENV PYTHONPATH=/opt/python-solver/GenericSolver/python:/opt/SEP/lib/python3.10:$PYTHONPATH
ENV LD_LIBRARY_PATH=/opt/SEP/lib
ENV PATH=$PATH:/opt/SEP/bin
ENV PYTHON=/usr/bin/python3.10
ENV PYTHONUSERBASE=/root/.local
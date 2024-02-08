FROM holoscan-base

ARG DEBIAN_FRONTEND=noninteractive

#
# Global Dependencies
#

RUN apt update

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN add-apt-repository ppa:tidewise/gstreamer-1.20
RUN apt update

RUN apt install -y git build-essential pkg-config git cmake
RUN apt install -y g++-11 gcc-11
RUN apt install -y python3-dev python3-pip
RUN python3 -m pip install meson ninja

#
# CyberEther Dependencies
#

RUN apt install -y libxkbcommon-dev

RUN apt update
RUN apt install -y libgstreamer1.0-dev gstreamer1.0-tools gstreamer1.0-libav
RUN apt install -y gstreamer1.0-plugins-base libgstreamer-plugins-bad1.0-dev 
RUN apt install -y libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev
RUN apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

RUN apt install -y libglfw3-dev
RUN apt install -y libvulkan-dev vulkan-validationlayers
RUN apt install -y python3-yaml

RUN apt install -y libsoapysdr-dev soapysdr-module-rtlsdr soapysdr-module-uhd 
RUN apt install -y soapysdr-module-airspy soapysdr-module-lms7 soapysdr-tools
RUN rm -fr /usr/lib/x86_64-linux-gnu/SoapySDR/modules0.7/libremoteSupport.so

RUN git clone https://github.com/KhronosGroup/glslang.git
RUN cd glslang \
    && git checkout 13.1.1 \
    && ./update_glslang_sources.py \
    && rm -fr build \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j8 \
    && make install

#
# BLADE Dependencies
#

RUN apt install -y liberfa-dev libhdf5-dev
RUN apt install -y libbenchmark-dev libgtest-dev
RUN python3 -m pip install numpy astropy pandas

#
# Workspace Setup
#

WORKDIR /workspace

#
# Clone, Build & Install CyberEther
#

RUN git clone https://github.com/luigifcruz/CyberEther.git
RUN cd CyberEther \
    && git checkout v1.0.0-alpha3 \
    && rm -fr build \
    && CC=gcc-11 CXX=g++-11 meson setup -Dbuildtype=debugoptimized build \
    && cd build \
    && ninja \
    && ninja install

#
# Clone, Build & Install BLADE
#

ENV NVCC_PREPEND_FLAGS='-ccbin g++-11'
RUN git clone https://github.com/luigifcruz/blade.git
RUN cd blade \
    && git submodule update --init --recursive \
    && rm -fr build \
    && CC=gcc-11 CXX=g++-11 meson setup -Dbuildtype=debugoptimized build \
    && cd build \
    && ninja \
    && ninja install
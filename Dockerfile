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
# Build Gstreamer (20.04 version too old)
#

RUN apt-get update && apt-get install -y \
    gtk-doc-tools \
    liborc-0.4-0 \
    liborc-0.4-dev \
    libvorbis-dev \
    libcdparanoia-dev \
    libcdparanoia0 \
    cdparanoia \
    libvisual-0.4-0 \
    libvisual-0.4-dev \
    libvisual-0.4-plugins \
    libvisual-projectm \
    vorbis-tools \
    vorbisgain \
    libopus-dev \
    libopus-doc \
    libopus0 \
    libopusfile-dev \
    libopusfile0 \
    libtheora-bin \
    libtheora-dev \
    libtheora-doc \
    libvpx-dev \
    libvpx-doc \
    bison \ 
    libgstreamer-plugins-base1.0-dev \ 
    libflac++-dev \
    libavc1394-dev \
    libraw1394-dev \
    libraw1394-tools \
    libraw1394-doc \
    libraw1394-tools \
    libtag1-dev \
    libtagc0-dev \
    libx264-dev \
    libx265-dev \
    libssl-dev \
    libwavpack-dev \
    wavpack \
    flex \ 
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git /gstreamer && \
    cd /gstreamer && \
    git checkout 1.20

RUN cd /gstreamer && \
    meson setup build --prefix=/usr --buildtype=release -Dgst-plugins-bad:nvcodec=enabled && \
    ninja -C build && \
    ninja -C build install

#
# CyberEther Dependencies
#

RUN apt update
RUN apt install -y libxkbcommon-dev
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
    && git checkout v1.0.0-alpha4 \
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

#
# Build & Install CyberBridge
#

COPY ./cyberbridge /workspace/cyberbridge
RUN cd cyberbridge \
    && rm -fr build \
    && CC=gcc-11 CXX=g++-11 meson setup -Dbuildtype=debugoptimized build \
    && cd build \
    && ninja \
    && ninja install

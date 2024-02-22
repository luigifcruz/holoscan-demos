# Holoscan Demos

The official Holoscan v0.6 docker image comes with Ubuntu 20.04 and CUDA 11.6. But BLADE and CyberEther requires at least CUDA 11.7 to work correcly. To fix this, we apply a patch to the `Dockerfile` distributed with the official image to update the base TensorRT docker image from 22.03 to 22.12. This will allow us to use CUDA 11.8 while still using Ubuntu 20.04 as the base image.

## Examples

- [ATA Spectrum Analyzer](./examples/ata_spectrum_analyzer/): Holoscan Advanced Networking and CyberEther effectively turning the Allen Telescope Array into a very large SDR.
- [IQ DSP](./examples/iq_dsp/): Hello world example on how to manipulate complex IQ stream within Holoscan and CyberEther.
- [Neural FM Radio](./examples/neural_fm_radio/): Example with a ONNX model directly demodulating a FM radio station from a SDR IQ stream.

## Build Development Image

### 1. Clone this repository
```
$ git clone https://github.com/luigifcruz/holoscan-demos
$ cd holoscan-demos
$ git submodule update --init --recursive
```

### 2. Patch the official Holoscan container
```
$ git apply patches/cuda_upgrade.patch 
```

### 3. Build official Holoscan container
```
$ cd holoscan-sdk
$ ./run setup
$ ./run build_image --platform linux/amd64 --gpu dgpu
$ cd ..
```

### 4. Build base container
```
$ docker build -t holoscan-base -f Dockerfile-base .
```

### 5. Build demo container
```
$ docker build -t holoscan-demo -f Dockerfile .
```

### 6. Run the demo container
```
$ nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)
$ sudo docker run -it --rm --net host --privileged --runtime=nvidia -u root \
    --device /dev/snd \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/huge:/mnt/huge \
    -v .:/workspace/demos \
    -v $nvidia_icd_json:$nvidia_icd_json:ro \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e DISPLAY=$DISPLAY \
    holoscan-demo
```

### 7. Compile the examples
This directory with the examples will be mounted at `/workspace/demos`. You can compile the examples with the following commands:
```
$ cd demos
$ CXX=g++-11 CC=gcc-11 meson -Dbuildtype=debugoptimized build
$ cd build
$ ninja
```

### 7. Fun!
Check each [example](#examples) README for further instructions.

## Feedback

- An easy way to install Holohub operators. I had to hack an installer for the Advanced Network operator.
- Holoscan complex number support. I guess this one is coming with an updated `libcudacxx`.
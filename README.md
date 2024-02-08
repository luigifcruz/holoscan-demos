# Holoscan Demos

The official Holoscan v0.6 docker image comes with Ubuntu 20.04 and CUDA 11.6. But BLADE and CyberEther requires at least CUDA 11.7 to work correcly. To fix this, we apply a patch to the `Dockerfile` distributed with the official image to update the base TensorRT docker image from 22.03 to 22.12. This will allow us to use CUDA 11.8 while still using Ubuntu 20.04 as the base image.

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

### 7. Fun!
This directory with the examples will be mounted at `/workspace/demos`. Check each example README for further instructions.

## Feedback

- An easy way to install Holohub operators. I had to hack an installer for the Advanced Network operator.
- Newer Ubuntu base image. Ubuntu 20.04 is a bit old and lacks C++20 support with the default GCC. A PPA is required for GCC-11.
- Updated CUDA version. Two year old CUDA 11.6 lacks C++20 support.
- Holoscan complex number support. I guess this one is coming with an updated `libcudacxx`.
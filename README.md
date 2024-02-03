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
$ 
```
THe official Holoscan v0.6 docker image comes with Ubuntu 20.04 and CUDA 11.6. But BLADE and CyberEther requires at least CUDA 11.7 to work correcly. To fix this, we apply a patch to the `Dockerfile` distributed with the official image to update the base TensorRT docker image from 22.03 to 22.12.

```
docker build -t holoscan-demo -f Dockerfile .
```
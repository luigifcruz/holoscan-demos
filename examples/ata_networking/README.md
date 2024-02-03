# Allen Telescope Array Advanced Networking Operator

Monitoring commands:
```
$ sudo mlnx_perf -i enp52s0f1np1
$ sudo tcpdump -xxxvvv -i enp52s0f1np1 -c 5
$ sudo tcpdump -w - -U | tee udp.dump | tcpdump -r -
$ sudo ibdev2netdev -v
$ sudo ibv_devinfo
```

Machine setup commands:
```
$ sudo cpufreq-set -g performance
$ sudo modprobe nvidia-peermem
$ sudo nvidia-smi -lmc 6500
```

Starting Docker container:
```
$ nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)
$ sudo docker run -it --rm --net host --privileged --runtime=nvidia -u root \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/huge:/mnt/huge \
    -v /home/sonata/holohub:/holohub \
    -v $nvidia_icd_json:$nvidia_icd_json:ro \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e DISPLAY=$DISPLAY \
    nvcr.io/k9lkna6dn1pr/advancednetworkop/advancednet:v0.6.0_x86_64
```

Automatic recompilation and execution of the application can be done with the following command:
```
$ cd /holohub/build
$ find ../applications/ | entr -r -s "make && /holohub/build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_tx_rx.yaml"
```
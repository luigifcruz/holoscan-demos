# Allen Telescope Array Advanced Networking Operator

## Usage

### 1. Build 
```
$ cd demos/examples/ata_networking/
$ CXX=g++-11 CC=gcc-11 meson -Dbuildtype=debugoptimized build
$ cd build
$ ninja
```

### 2. Setup machine
This example requires `nvidia-peermem` to be active. It's also best practice to disable dynamic clock in the CPU. This can be done by executing these commands in the **host machine**:
```
$ sudo cpufreq-set -g performance
$ sudo modprobe nvidia-peermem
$ sudo nvidia-smi -lmc 6500  # for A4000
```

### 3. Check configuration
Check if the target network card is correct in the configuration YAML file.

### 4. Run
```
$ ./main ../default.yaml
```

## Useful Commands 

### Monitoring commands
```
$ sudo mlnx_perf -i enp52s0f1np1
$ sudo tcpdump -xxxvvv -i enp52s0f1np1 -c 5
$ sudo tcpdump -w - -U | tee udp.dump | tcpdump -r -
$ sudo ibdev2netdev -v
$ sudo ibv_devinfo
```

### Development commands
Automatic recompilation and execution of the application can be done with the following command:
```
$ cd /holohub/build
$ find ../applications/ | entr -r -s "make && /holohub/build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_tx_rx.yaml"
```
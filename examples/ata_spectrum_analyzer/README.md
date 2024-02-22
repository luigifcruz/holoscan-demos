# Allen Telescope Array Spectrum Analyzer
This demo will interface with the Allen Telescope Array (ATA) network via the Holoscan Advanced Network Operator (ANO) to receive and process data from the ATA network. The data will be processed using the Breakthrough Listen Accelerated DSP Engine (BLADE) to produce beamformed data. The beamformed data will be sent to CyberEther where it will be processed for visualization.

## Usage

### 1. Setup machine
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
$ cd examples/ata_spectrum_analyzer
$ ./main
```
This will start the application and it will start receiving data from the ATA network. The CyberEther interface can be accessed using a local CyberEther instance with the Remote Block. Check out the [CyberEther documentation](https://github.com/luigifcruz/CyberEther?tab=readme-ov-file#remote-interface) for more information. To stop the application, press `Ctrl+C`.

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
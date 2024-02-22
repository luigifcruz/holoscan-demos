# Machine Learning Based FM Demodulation
This demo will receive an FM radio station from a SDR using a SoapySDR Holoscan operator and demodulate the RAW I/Q data using a ONNX model. The demodulated audio will be sent to CyberEther for visualization and playback.

## Usage

### 1. Run
```
$ cd examples/neural_fm_radio
$ ./main
```
This will start the application. The CyberEther interface can be accessed using a local CyberEther instance with the Remote Block. Check out the [CyberEther documentation](https://github.com/luigifcruz/CyberEther?tab=readme-ov-file#remote-interface) for more information. To stop the application, press `Ctrl+C`.

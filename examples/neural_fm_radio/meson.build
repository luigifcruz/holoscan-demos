custom_target(
    'cyberether.yml',
    input: 'cyberether.yml',
    output:  'cyberether.yml',
    command: ['cp', '@INPUT@', '@OUTPUT@'],
    install: false,
    build_by_default: true
)

custom_target(
    'holoscan.yaml',
    input: 'holoscan.yaml',
    output:  'holoscan.yaml',
    command: ['cp', '@INPUT@', '@OUTPUT@'],
    install: false,
    build_by_default: true
)

custom_target(
    'cursednet.onnx',
    input: 'cursednet.onnx',
    output:  'cursednet.onnx',
    command: ['cp', '@INPUT@', '@OUTPUT@'],
    install: false,
    build_by_default: true
)

executable(
    'neural_fm_radio', 
    'main.cc',
    dependencies: [
        dependency('cyberbridge-holoscan'),
        dependency(
            'holoscan', 
            modules: [
                'holoscan::ops::inference',
            ]
        ),
        dependency('soapysdr', 'SoapySDR'),
    ],
    install: true,
)
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

executable(
    'iq_dsp', 
    'main.cc',
    dependencies: [
        dependency('cyberbridge-holoscan'),
    ],
    install: true,
)
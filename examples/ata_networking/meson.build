custom_target(
    'default.yaml',
    input: 'default.yaml',
    output:  'default.yaml',
    command: ['cp', '@INPUT@', '@OUTPUT@'],
    install: false,
    build_by_default: true
)

src_lst = [
    'main.cc',
    'defragmentation_kernel.cu',
]

executable(
    'ata_networking', 
    src_lst,
    dependencies: [
        dependency('cuda', modules: ['cudart', 'culibos']),
        dependency(
            'holoscan', 
            modules: [
                'holoscan::core',
                'holoscan::logger',
            ]
        ),
        dependency(
            'AdvancedNetwork', 
            method : 'cmake',
            modules: [
                'holoscan::ops::advanced_network_common',
                'holoscan::ops::advanced_network_rx',
            ]
        ),
    ],
    install: true,
)
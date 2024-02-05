project(
    'ATA Networking',
    ['cpp', 'cuda'],
    version: '0.0.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++17',
    ]
)

src_lst = [
    'main.cc',
    'defragmentation_kernel.cu',
]

executable(
    'main', 
    src_lst,
    dependencies: [
        dependency('cuda', modules: ['cudart', 'culibos']),
        dependency(
            'holoscan', 
            modules: [
                'holoscan::core', 
                'holoscan::logger', 
                'holoscan::infer',
                'holoscan::ops::inference',
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
        dependency('jetstream'),
        dependency('soapysdr', 'SoapySDR'),
    ],
    install: true,
    # Meson is not setting this for some reason.
    install_rpath: '/opt/nvidia/holoscan/lib/',
    build_rpath: '/opt/nvidia/holoscan/lib/',
)
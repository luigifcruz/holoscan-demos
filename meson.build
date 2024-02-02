project(
    '{Holoscan + BLADE + CyberEther} Demos',
    ['cpp'],
    version: '0.0.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

executable(
    'main', 
    'main.cc',
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
        dependency('blade'),
        dependency('jetstream'),
        dependency('soapysdr', 'SoapySDR'),
    ],
    install: true,
    # Meson is not setting this for some reason.
    install_rpath: '/opt/nvidia/holoscan/lib/',
    build_rpath: '/opt/nvidia/holoscan/lib/',
)
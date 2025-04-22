from setuptools import find_packages, setup

package_name = 'rob8_861'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools', 'rclpy', 'rob8_interfaces'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='70377798+Jakob-Jorgensen@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tcp_server = rob8_861.tcp_server:main'
        ],
    },
)

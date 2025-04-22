from setuptools import setup, find_packages

package_name = 'gg_cnn'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # auto-detect packages
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',  # Standard
        'torch',        # For GG_CNN (deep learning)
        'numpy',        # For numerical operations
        'scikit-image', # For image processing
        'opencv-python', # For OpenCV functions 
        'rob8_interfaces', # For costom messge 
    ],
    zip_safe=True,
    maintainer='max',
    maintainer_email='70377798+Jakob-Jorgensen@users.noreply.github.com',
    description='GG_CNN for Grasp Detection with RealSense Camera in ROS 2',
    license='TODO',
    entry_points={
        'console_scripts': [
            'gg_cnn_image_processing = gg_cnn.gg_cnn_interface:main',  # Updated entry point to match your script
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'gg_CNN'

setup(
    name=package_name,
    version='0.0.1',  # Updated version
    packages=find_packages(exclude=['test']),  # Automatically find and include packages, excluding 'test' folder
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
        'opencv-python' # For OpenCV functions
    ],
    zip_safe=True,  # Set this to False if the package can't run in a zipped form
    maintainer='max',  # Your name or organization
    maintainer_email='70377798+Jakob-Jorgensen@users.noreply.github.com',  # Your email
    description='GG_CNN for Grasp Detection with RealSense Camera in ROS 2',  # Package description
    license='Apache 2.0',  # License for the package
    tests_require=['pytest'],  # Testing framework
    entry_points={
        'console_scripts': [
            'gg_cnn_image_processing = gg_CNN.gg_cnn_interface:main',  # Updated entry point to match your script
        ],
    },
)

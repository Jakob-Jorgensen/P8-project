from setuptools import find_packages, setup

package_name = 'vlm'

setup(
    name=package_name,
    version='0.0.1',  # Updated version
    packages=find_packages(where='vlm/vlm', exclude=['test']),  # Automatically find packages in the gg_cnn/gg_cnn directory
    package_dir={'src': 'vlm'},  # Correctly point to the inner gg_cnn directory for Python files
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',  # Standard
        'torch',        # For running the models
        'numpy',        # For numerical operations
        'scikit-image', # For image processing
        'opencv-python' # For OpenCV functions
    ],
    zip_safe=True,  # Set this to False if the package can't run in a zipped form
    maintainer='max',  # Your name or organization
    maintainer_email='70377798+Jakob-Jorgensen@users.noreply.github.com',  # Your email
    description='VLM package for P8 project',  # Package description
    license='todo',  # License for the package
    tests_require=['pytest'],  # Testing framework
    entry_points={
        'console_scripts': [
            'vlm_interface = vlm.vlm_interface:main',  # Updated entry point to match your script
        ],
    },
)

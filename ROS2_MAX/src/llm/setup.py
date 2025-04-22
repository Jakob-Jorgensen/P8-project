from setuptools import setup, find_packages

package_name = 'llm'

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
        'setuptools',
        'torch',
        'numpy',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='max',
    maintainer_email='70377798+Jakob-Jorgensen@users.noreply.github.com',
    description='LLM package for P8 project',
    license='TODO',
    entry_points={
        'console_scripts': [
            'llm_interface = llm.llm_interface:main',
        ],
    },
)

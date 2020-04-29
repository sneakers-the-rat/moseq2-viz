from setuptools import setup, find_packages
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')


setup(
    name='moseq2-viz',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.2.2',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['tqdm==4.40.0', 'matplotlib==3.1.2', 'click==7.0', 'dtaidistance==1.2.3', 'scikit-learn==0.22',
                      'ruamel.yaml==0.16.5', 'seaborn==0.9.0', 'opencv-python==4.1.2.30', 'psutil==5.6.7',
                      'pandas==0.25.3', 'networkx==2.4', 'numpy==1.17.4', 'h5py==2.10.0', 'cytoolz==0.10.1'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)

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
    version='0.2.5',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['tqdm', 'matplotlib', 'click', 'dtaidistance', 'scikit-learn',
                      'ruamel.yaml', 'seaborn', 'opencv-python', 'psutil',
                      'pandas', 'networkx', 'numpy', 'cytoolz'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)

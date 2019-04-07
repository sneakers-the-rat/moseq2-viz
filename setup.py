from setuptools import setup, find_packages

setup(
    name='moseq2-viz',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.1.3',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['tqdm', 'matplotlib', 'click', 'dtaidistance', 'sklearn',
                      'ruamel.yaml>=0.15.0', 'seaborn', 'opencv-python',
                      'pandas', 'networkx', 'numpy==1.14.5', 'cytoolz'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)

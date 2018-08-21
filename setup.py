from setuptools import setup

setup(
    name='moseq2-viz',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.1.0',
    platforms=['mac', 'unix'],
    install_requires=['tqdm', 'matplotlib', 'click', 'ruamel.yaml<=0.15.0',
                      'seaborn', 'opencv-python', 'pandas', 'networkx'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)

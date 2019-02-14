from setuptools import setup

setup(
    name='ipylmfit',
    version='0.1',
    author='William',
    packages=['ipylmfit'],
    install_requires=['ipympl', 'lmfit', 'matplotlib', 'ipywidgets', 'numpy', 'IPython'],
    zip_safe=False
)

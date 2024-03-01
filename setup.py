from setuptools import find_packages, setup

setup(
    name='src',
    version='0.1.0',
    description='Prediction of reference evapotranspiration (ET0) using the POWER NASA dataset',
    author='Francesco Pasanisi',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

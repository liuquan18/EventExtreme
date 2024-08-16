from setuptools import setup, find_packages

setup(
    name="eventextreme",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
    ],
    author="Quan Liu",
    author_email="quan.liu@mpimet.mpg.de",
    description="A package for extracting extreme events",
    license="MIT",
    python_requires=">=3.10",
)

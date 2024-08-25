from setuptools import setup, find_packages

setup(
    name="eventextreme",
    version="0.6.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
    ],
    author="Quan Liu",
    author_email="quan.liu@mpimet.mpg.de",
    description="A package for extracting extreme events",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liuquan18/EventExtreme",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

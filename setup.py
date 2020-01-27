import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="zsvision",
    version="0.1.2",
    author="Samuel Albanie",
    description="Python utilities for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/albanie/zsvision",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

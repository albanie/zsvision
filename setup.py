"""
coverage run --source=. -m py.test
python3 setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
"""
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="zsvision",
    version="0.7.7",
    author="Samuel Albanie",
    description="Python utilities for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/albanie/zsvision",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "msgpack",
        "msgpack_numpy",
        "typeguard",
        "scipy",
        "mergedeep",
        "humanize",
        "matplotlib",
        "beartype>=0.3.2",
        "hickle>=4.0.0",
        "pyyaml",
    ],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ],
)

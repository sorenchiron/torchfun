import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchure",
    version="0.0.14",
    author="CHEN Si Yu",
    author_email="sychen@zju.edu.cn",
    description="A collection of small functions that supplements torch functionality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sorenchiron/torchfun",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License"
    ),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib']
)
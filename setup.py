import setuptools
import pickle
import os


def increase_version_number():
    if os.path.exists('version'):
        version = pickle.load(open('version','rb'))
    else:
        version = '0.0.0'
    print('previous version',version)
    release,updates,fixes = version.split('.')
    fixes = str(int(fixes)+1)
    version = '.'.join([release,updates,fixes])
    pickle.dump(version,open('version','wb'))
    
    print('new version',version)
    return version

def write_version(v):
    with open('torchfun/version','w') as f:
        f.write(v)

version = increase_version_number()
write_version(version)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchfun",
    version=version,
    author="CHEN Si Yu",
    author_email="sychen@zju.edu.cn",
    description="A collection of small functions that supplements torch functionality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sorenchiron/torchfun",
    packages=setuptools.find_packages(),
    include_package_data=True,
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
        'matplotlib',
        'Pillow',
        'scipy',
        'tqdm']
)
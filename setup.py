import setuptools
import pickle
import os


def increase_version_number():
    if os.path.exists('version'):
        version = open('version','r')
    else:
        version = '0.0.0'
    print('previous version',version)
    release,updates,fixes = version.read().split('.')
    updates_increase=release_increase=0
    if '+' in fixes:
        fixes = '-1'
        updates_increase = 1
    if '+' in updates:
        updates = '0'
        fixes = '-1'
        release_increase = 1

    fixes = str(int(fixes)+1)
    updates = str(int(updates)+updates_increase)
    release = str(int(release)+release_increase)

    version.close()
    new_version = open('version','w')
    version = '.'.join([release,updates,fixes])
    new_version.write(version)
    print('new version',version)
    return version

def write_version(v):
    with open('torchfun/version','w') as f:
        f.write(v)

def write_install(v):
    fname = 'torchfun-%s-py3-none-any.whl' % (v)
    fpath = os.path.join('dist',fname)
    with open('local_install.bat','w') as f:
        f.write('pip install '+fpath)

version = increase_version_number()
write_version(version)
write_install(version)

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
    package_data={
                '':['version']
                },
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
        'matplotlib>=3.0.0',
        'Pillow',
        'scipy',
        'tqdm',
        'psutil',
        'tensorboardX<=1.5',
        'imageio',
        'img2pdf'],
    entry_points={
        'console_scripts': [
            'imgformat = torchfun.tools.imgformat:main'
        ],
        'gui_scripts': [
        ]
    }
)
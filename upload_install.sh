#!/bin/bash
rm -rf ./dist/* 
python3 ./setup.py sdist bdist_wheel
twine upload dist/*
cat local_install.bat | bash

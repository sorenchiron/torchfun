rm -rf ./docs/*
cd torchfun/doc
rm modules.rst
rm torchfun.rst
sphinx-apidoc -o . ..
make.bat html
del .\docs\* -recurse -force
cd .\torchfun\doc\
del .\modules.rst
del .\torchfun.rst
sphinx-apidoc.exe -o . ..
.\make.bat html

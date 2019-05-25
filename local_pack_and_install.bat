del -Force -Recurse .\dist\* 
python .\setup.py sdist bdist_wheel
local_install.bat
pause
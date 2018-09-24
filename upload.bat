del -Force -Recurse .\dist\* 
python .\setup.py sdist bdist_wheel
twine upload dist/*
pause
del -Force -Recurse .\dist\* 
python .\setup_legacy.py sdist bdist_wheel
twine upload dist/*
pause
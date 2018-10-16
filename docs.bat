call .\make_docs.bat
cd ..\..\
echo continuing
copy  .\torchfun\doc\_build\html\* .\docs\  /Y
echo done!
pause
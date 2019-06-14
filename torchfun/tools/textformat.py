# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2019
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 


from __future__ import print_function
from __future__ import division
import argparse
from ..torchfun import safe_open
import os
from glob import glob
from sys import argv

parser = argparse.ArgumentParser(description='Text File Format Transformer')

parser.add_argument('files', metavar='FILE_PATTERN', type=str, nargs='+',
                        help='files to be transformed, can be like xx/yy/z.txt, or like xx/yy/*.png')
parser.add_argument('-t','--type',dest='type', type=str, default=None, help='output text encoding type')
parser.add_argument('-d','--dir',dest='dir', type=str, default=None, help='dir to store output files, default: same dir as the input.')
parser.add_argument('-v','--verbose',dest='verbose', action='store_true', default=False, help='print the encodings for each files')
parser.add_argument('-p','--print',dest='print', action='store_true', default=False, help='print known encodings')

all_encodings = ['utf_8','utf_16','utf_16_be','utf_16_le','utf_7','gb2312','gbk','gb18030','ascii','big5','big5hkscs','cp037','cp424','cp437','cp500','cp737','cp775','cp850','cp852','cp855','cp856','cp857','cp860','cp861','cp862','cp863','cp864','cp865','cp866','cp869','cp874','cp875','cp932','cp949','cp950','cp1006','cp1026','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','cp1255','cp1256','cp1257','cp1258','euc_jp','euc_jis_2004','euc_jisx0213','euc_kr','hz','iso2022_jp','iso2022_jp_1','iso2022_jp_2','iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext','iso2022_kr','latin_1','iso8859_2','iso8859_3','iso8859_4','iso8859_5','iso8859_6','iso8859_7','iso8859_8','iso8859_9','iso8859_10','iso8859_13','iso8859_14','iso8859_15','johab','koi8_r','koi8_u','mac_cyrillic','mac_greek','mac_iceland','mac_latin2','mac_roman','mac_turkish','ptcp154','shift_jis','shift_jis_2004','shift_jisx0213']
all_encodings = sorted(all_encodings,reverse=True)
# all in lower case

def textformat():
    opt=parser.parse_args(argv[1:])

    if opt.print:
        print('supported encodings are:')
        for enc in all_encodings:
            print(enc)
        print('supported encodings are above selections.')
        return 0

    if not opt.type:
        print('please specify output type by -t or --type possible values:',','.join(all_encodings))
        print('blank type will cause the program to only print the file encodings')
    else:
        opt.type = opt.type.lower().replace('-','_')

    more_files = []
    files = []
    for fpath in opt.files:
        if '*' in fpath:
            more_files.extend(glob(fpath))
        else:
            files.append(fpath)
    all_fpaths = list(set(more_files+files))
    filenumbers = len(all_fpaths)
    print('===============================')
    for i,textpath in enumerate(all_fpaths):
        i = i+1
        if not os.path.isfile(textpath):
            print(textpath,'not a file')
            continue

        f,enc = safe_open(textpath,return_encoding=True,verbose=False)
        if f is None:
            print('Encoding cannot be determined, skipping',textpath)
        elif enc == opt.type:
            if opt.verbose:
                print(textpath,'is already encoded using',enc)
        else:
            if opt.type is None:
                print(textpath,enc)
            else:
                text = f.read()
                f.close()
                with open(textpath,'w',encoding=opt.type) as f:
                    f.write(text)
                    if opt.verbose:
                        print(textpath,enc,'==>',opt.type)

    return 0

def main():
    return textformat()

if __name__ == '__main__':
  main()
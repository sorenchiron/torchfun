# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import print_function
from __future__ import division
import argparse,re,os
import shutil as fs
from ..torchfun import safe_open

parser = argparse.ArgumentParser(description='latex引用图片提取打包器')
parser.add_argument('--input', dest='input', type=str, default='', help='待处理文件路径')
parser.add_argument('--out', dest='out', type=str, default='', help='提取输出到目标目录')
parser.add_argument('--dry', dest='dry', action='store_true', default=False, help='只打印行为，不执行任何更改')

args = parser.parse_args()




def force_exist(dirname):
    if dirname == '' or dirname == '.':
        return True
    top = os.path.dirname(dirname)
    force_exist(top)
    if not os.path.exists(dirname):
        print('creating',dirname)
        os.makedirs(dirname)
        return False
    else:
        return True

def main():
    content = safe_open(args.input,'r').readlines()

    comment_filter_pattern = re.compile('^\s*\%')

    ignored = [c for c in content if comment_filter_pattern.match(c) is not None]
    #for i in ignored:
    #    print('ignored:',i)

    content = [c for c in content if comment_filter_pattern.match(c) is None]

    content = ''.join(content)

    file_pattern = re.compile('includegraphics.*?\{\s*?(.+?\.(?:jpg|png|pdf|eps))\s*?\}')
    res = file_pattern.findall(content,re.IGNORECASE|re.DOTALL|re.MULTILINE)
    src_base_path = os.path.dirname(args.input)
    src_file_paths = [os.path.join(src_base_path,p) for p in res]

    dest_base_path = args.out
    dest_dirnames = [os.path.dirname(p) for p in res]
    dest_dirpaths = [os.path.join(dest_base_path,n) for n in dest_dirnames]
    dest_file_paths = [os.path.join(dest_base_path,p) for p in res]

    print('found',len(res),'images in',args.input)
    if args.dry:
        for src in res:
            print('found',src)
        return

    for dest_dirpath in dest_dirpaths:
        force_exist(dest_dirpath)

    for src,dest in zip(src_file_paths,dest_file_paths):
        fs.copy(src,dest)

    print('done')



if __name__ == '__main__':
    main()


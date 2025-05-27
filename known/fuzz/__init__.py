
__doc__=r"""fuzz module"""

import os
from ..basic import Fuzz as F

class Actions:
    
    @staticmethod
    def info(*paths):
        #os.path.exists(path), os.path.isdir(path), os.path.isfile(path), os.path.islink(path)
        for path in paths:
            e,d,f,l = F.ExistInfo(path)
            c1 = '✅' if e else '❌'
            c2 = '📁' if d else ('📄' if f else '❓')
            c3 = '🔗' if l else '🔹' if e else '❓'
            #vb.strD_(dict(exists=e, isDir=d, isFile=f, isLink=l), cep="\t", caption=path)
            print (f'{c1} {c2} {c3} {path}')


    @staticmethod
    def scan(*args):
        assert len(args)==3, f'expected 3 args exactly but got {len(args)} :: {args}'
        L = F.Scan(path=args[0], exclude_hidden=bool(args[1]), include_size=bool(args[2]), include_extra=False)
        for i,l in enumerate(L, 1):
            name, path, isdir, isfile, islink, size, *E = l
            parent, fname, ext = E if E else (None, None, None)

            c1 = '📁' if isdir else ('📄' if isfile else '❓')
            c2 = '🔗' if islink else '🔹' 
            c3 =  f' ~ {size} Bytes' if isfile else ''
            print(f"[{i}] {c1} {c2} {name} @ {path}{c3}")
            #print(f'[{i}]\t\t{l}')


    @staticmethod
    def rescan(*args):
        assert len(args)==3, f'expected 3 args exactly but got {len(args)} :: {args}'
        L = F.ReScan(path=args[0], exclude_hidden=bool(args[1]), include_size=bool(args[2]), include_extra=True)
        for i,l in enumerate(L, 1):
            name, path, isdir, isfile, islink, size, *E = l
            parent, fname, ext = E if E else (None, None, None)

            c1 = '📁' if isdir else ('📄' if isfile else '❓')
            c2 = '🔗' if islink else '🔹' 
            c3 =  f' ~ {size} Bytes' if isfile else ''
            c4 = f'"{fname}"."{ext}" @ {parent}'
            #print(f"[{i}] {c1} {c2} {name} {c4} @ {path}{c3}")
            print(f"[{i}] {c1} {c2} {c4} {c3}")
            #print(f'[{i}]\t\t{l}')




__doc__=r"""pix execute"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] Can not import {__name__}:{__file__}')
#-----------------------------------------------------------------------------------------

import os, json, argparse
from . import Actions

avaliable_actions = [k for k in Actions.__dict__ if not k.startswith('__')]
#-----------------------------------------------------------------------------------------

def _read_fl_from_text(path):
    with open(path, 'r') as f: l = [ os.path.abspath(f'{s}') for s in f.read().split('\n') if s ]
    return l
def _read_fl_from_json(path):
    with open(path, 'rb') as f: l = json.load(f)
    return l
def _read_fl_from_linux(F): # parses --files="%F"
    Fl = [fi.strip() for fi in F.split("'/")]
    Fr = [os.path.abspath(f'/{fl[:-1]}'.replace("'\\''","'")) for fl in Fl if fl] 
    return Fr
def _read_fl(parsed_put, parsed_io, check=False):
    if parsed_put:
        _put = os.path.abspath(parsed_put)
        if not parsed_io:           _puts = [_put] 
        elif parsed_io == 't':   _puts =_read_fl_from_text(_put)
        elif parsed_io == 'j':   _puts =_read_fl_from_json(_put)
        elif parsed_io == 'l':   _puts =_read_fl_from_linux(_put)
        else:                       _puts = [] 
    else:                           _puts = []
    if check: _puts = [p for p in _puts if os.path.isfile(p)]
    return _puts


#-----------------------------------------------------------------------------------------

# actions = new, crop, extend, flip, rotate, convert
parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str, default='',   help=f"(str) one of the static-methods inside the Actions class, can be - {avaliable_actions}")
parser.add_argument('--args',   type=str, default='',   help="(str) csv args accepted by the specified action - each action takes different args")
parser.add_argument('--input',  type=str,   default='', help='(str) input  image-file or a text/json-file containing multiple input image-file names') 
parser.add_argument('--output',  type=str,  default='', help='(str) output image-file or a text/json-file containing multiple output image-file names') 
parser.add_argument('--files',  type=str,   default='', help='(str) multiple input image-file names - for custom action -- works only with --io=linux') 
parser.add_argument('--io',      type=str,  default='', help="(str) can be 'text' or 'json' or 'linux' - keep blank to io as 'image' - used if providing input/output file-names in a text/json file")
parser.add_argument('--verbose', type=int,  default=0,  help="(int) verbose level - 0 or 1")

parsed = parser.parse_args()

# ---------------------------------------------------------------------------------
_verbose = int(parsed.verbose)
_action = f'{parsed.action}'
if not _action: exit(f'[!] Action not provided')
if not hasattr(Actions, _action): exit(f'[!] Action [{_action}] not found')
_action = getattr(Actions, _action)
# ---------------------------------------------------------------------------------
_args = f'{parsed.args}'.split(',')
# ---------------------------------------------------------------------------------
_io = f'{parsed.io}'.lower()[0]
if _io == 'l':
    _inputs =   _read_fl(f'{parsed.files}',  _io, check=True) # assume existing files are passed
    _outputs =  _inputs
else:
    _inputs =   _read_fl(f'{parsed.input}',  _io, check=True) # keep only existing files
    _outputs =  _read_fl(f'{parsed.output}', _io, check=False)
    if not _outputs: _outputs = _inputs # if outputs are not provided, overwrite inputs
    if _inputs: assert len(_inputs) == len(_outputs), f'Mismatch inputs and outputs' # if inputs were provided, outputs must match them

# ---------------------------------------------------------------------------------
_action(_inputs, _outputs, _args, _verbose)
# ---------------------------------------------------------------------------------

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
def _read_fl(parsed_put, parsed_io):
    if parsed_put:
        _put = os.path.abspath(parsed_put)
        if not parsed_io:           _puts = [_put] 
        elif parsed_io[0] == 't':   _puts =_read_fl_from_text(_put)
        elif parsed_io[0] == 'j':   _puts =_read_fl_from_json(_put)
        else:                       _puts = [] # raise TypeError(f'Invalid io type [{parsed_io}]')
    else:                           _puts = []
    return _puts

#-----------------------------------------------------------------------------------------

# actions = new, crop, extend, flip, rotate, convert
parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str, default='',   help=f"(str) one of the static-methods inside the Actions class, can be - {avaliable_actions}")
parser.add_argument('--args',   type=str, default='',   help="(str) csv args accepted by the specified action - each action takes different args")
parser.add_argument('--input',  type=str,   default='', help='(str) input  image-file or a text/json-file containing multiple input image-file names') 
parser.add_argument('--output',  type=str,  default='', help='(str) output image-file or a text/json-file containing multiple output image-file names') 
parser.add_argument('--io',      type=str,  default='', help="(str) can be 'text' or 'json' - keep blank to io as 'image' - used if providing input/output file-names in a text/json file")
parser.add_argument('--verbose', type=int,  default=0,  help="(int) verbose level - 0 or 1")
#parser.add_argument('--help')
parsed = parser.parse_args()

#print(parsed.help)
# python3.11 -m known.pix --input=ava.png --output=ava.jpg --io= --action=convert --args=

#-----------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
_verbose = int(parsed.verbose)
_action = f'{parsed.action}'
if not _action: exit(f'[!] Action not provided')
if not hasattr(Actions, _action): exit(f'[!] Action [{_action}] not found')
_action = getattr(Actions, _action)
# ---------------------------------------------------------------------------------
_args = f'{parsed.args}'.split(',')
# ---------------------------------------------------------------------------------
_inputs = _read_fl(f'{parsed.input}',  f'{parsed.io}'.lower())
_inputs = [s for s in _inputs if os.path.isfile(s)] # keep only existing files
_outputs = _read_fl(f'{parsed.output}', f'{parsed.io}'.lower())
if not _outputs: _outputs = _inputs # if outputs are not provided, overwrite inputs
if _inputs: assert len(_inputs) == len(_outputs), f'Mismatch inputs and outputs' # if inputs were provided, outputs must match them
else:       pass # print(f'No input files to process')
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
_action(_inputs, _outputs, _args, _verbose)
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

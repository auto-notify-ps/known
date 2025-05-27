__doc__=r"""fuzz execute"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] Can not import {__name__}:{__file__}')
#-----------------------------------------------------------------------------------------

import os, argparse
from . import Actions

#-----------------------------------------------------------------------------------------
avaliable_actions = [k for k in Actions.__dict__ if not k.startswith('__')]
# actions = new, crop, extend, flip, rotate, convert
parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str, default='',   help=f"(str) one of the actions, can be - {avaliable_actions}")
parser.add_argument('--args',   type=str, default='',   help="(str) csv args accepted by the specified action - each action takes different args")
parser.add_argument('--verbose', type=int,  default=0,  help="(int) verbose level - 0 or 1")

parsed = parser.parse_args()

# # ---------------------------------------------------------------------------------
# _verbose = int(parsed.verbose)
_action = f'{parsed.action}'
if not _action: exit(f'[!] Actions not provided')
if not hasattr(Actions, _action): exit(f'[!] Actions [{_action}] not found')
_action = getattr(Actions, _action)
# # ---------------------------------------------------------------------------------
_args = f'{parsed.args}'.split(',')
# # ---------------------------------------------------------------------------------
# if not parsed.io: _io = 'i'
# else: _io = f'{parsed.io}'.lower()[0]
# if _io == 'l':
#     _inputs =   _read_fl(f'{parsed.files}',  _io, check=True) # assume existing files are passed
#     _outputs =  _inputs
# else:

#     _inputs =   _read_fl(f'{parsed.input}',  _io, check=True) # keep only existing files
#     _outputs =  _read_fl(f'{parsed.output}', _io, check=False)
    
#     if not _outputs: _outputs = _inputs # if outputs are not provided, overwrite inputs
#     if _inputs: assert len(_inputs) == len(_outputs), f'Mismatch inputs and outputs' # if inputs were provided, outputs must match them

# # ---------------------------------------------------------------------------------
_action(*_args)

# # ---------------------------------------------------------------------------------

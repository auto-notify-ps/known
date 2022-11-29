
# * * * * * * * * * * * * * * * * * * * * * * * * * *
# known/basic.py
# * * * * * * * * * * * * * * * * * * * * * * * * * *
#import known.basic as basic
# Aliased
from known.basic import now, fdate, pdate, fake
# Path related
from known.basic import pjs, pj
# Printing
from known.basic import strA, strD, strU, show, showX
# Misc
from known.basic import REMAP, JSON

# * * * * * * * * * * * * * * * * * * * * * * * * * *


# * * * * * * * * * * * * * * * * * * * * * * * * * *
# known/hyper
# * * * * * * * * * * * * * * * * * * * * * * * * * *
#import known.hyper as hyper
from known.hyper.core import LOGGER, FAKER, ESCAPER
from known.hyper.html import HTAG, HARSER, from_file, from_web, to_file
from known.hyper.md import MarkDownLogger
from known.hyper.mu import MarkUpLogger
# * * * * * * * * * * * * * * * * * * * * * * * * * *



# * * * * * * * * * * * * * * * * * * * * * * * * * *
# known/deep
# * * * * * * * * * * * * * * * * * * * * * * * * * *
#import known.deep as deep
#import known.deep.common as common
from known.deep.common import build_dense_sequential, save_w8s, load_w8s, make_clone, make_clones

#import known.deep.mlp as mlp
from known.deep.mlp import MLP, MLPn


# * * * * * * * * * * * * * * * * * * * * * * * * * *
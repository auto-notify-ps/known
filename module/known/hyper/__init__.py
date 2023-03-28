# hyper
__doc__=r"""

Hypertext based tools, contains mark-up and mark-down loggers and other tools for parsing html files

:py:mod:`known/hyper/__init__.py`
"""
__all__ = [
    'LOGGER', 'FAKER', 'ESCAPER',
    'HTAG', 'HARSER', 'from_file', 'from_web', 'to_file',
    'MarkDownLogger', 'MarkUpLogger'
]

from .core import LOGGER, FAKER, ESCAPER
from .html import HTAG, HARSER, from_file, from_web, to_file
from .md import MarkDownLogger
from .mu import MarkUpLogger


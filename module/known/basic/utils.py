__doc__=r"""
===========
Utilities
===========

Contains shorthand/helper functions. Misc objects.

"""

__all__ = [
    'pjs', 'pj', 'pname', 'pext', 'psplit',
    'FAKE',
]
#-----------------------------------------------------------------------------------------------------
import os.path
#-----------------------------------------------------------------------------------------------------


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Path related functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def pjs(*paths) -> str:
    r""" Path Joins : shorthand for `os.path.join`, can take multiple paths """
    res = ''
    for p in paths: res = os.path.join(res, p)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pj(path, sep='/') -> str: 
    r""" Path Join : shorthand for `os.path.join`.
    similar to `pjs` but takes a single string instead of multiple args """
    return pjs(*path.split(sep))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pname(path, sep='.'): 
    r""" retuns the path except file extension """
    return path[0:path.rfind(sep)]
def pext(path, sep='.'): 
    r""" retuns the extension from a path """
    return path[path.rfind(sep):]
def psplit(path, sep='.'): 
    r""" splits the path into name and extension
    
    > This may be used to create copies of a file by adding a suffix to its name witout changing extension
    """
    return (path[0:path.rfind(sep)], path[path.rfind(sep):])


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Misc
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class FAKE:
    r""" an object with given members (dict) """
    def __init__(self, **members) -> None:
        for k,v in members.items(): setattr(self, k, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



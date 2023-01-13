#-----------------------------------------------------------------------------------------------------
import os.path
#-----------------------------------------------------------------------------------------------------


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Path related functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pjs(*paths) -> str:
    """ Path Joins : joins multiple dirs/files in args using os.path.join """
    res = ''
    for p in paths: res = os.path.join(res, p)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pj(path, sep='/') -> str: 
    """ Path Join : similar to pjs but takes a string instead of individual args """
    return pjs(*path.split(sep))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pname(path, sep='.'): return path[0:path.rfind(sep)]
def pext(path, sep='.'): return path[path.rfind(sep):]
def psplit(path, sep='.'): return (path[0:path.rfind(sep)], path[path.rfind(sep):])
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Misc
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class FAKE:
    """ an object with given members (dict) """
    def __init__(self, **members) -> None:
        for k,v in members.items(): setattr(self, k, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



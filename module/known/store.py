__doc__=r"""
:py:mod:`known/store.py`
"""
import os

__all__ = ['Node', 'Store']

def isValidNodeName(k:str): return ((not (k.startswith('__') and k.endswith('__'))) and k.isidentifier())

class Node:
    r""" Node Object - represents a directory tree.
    Each node has:
    * ```__path__``` which specifies absolute path of Node's root
    * ```__files__``` a list of file names in the directory
    * sub-directories are set as attributes of Node type
    
    .. note:: calling a Node object with no args returns it full path, if any argument is provides, uses ```os.join``` to append it

    .. note:: Not to be used directly, these are attributes on ```Store``` objects.
    """

    def __init__(self, path, members=None) -> None:
        if members:
            for k,v in members.items(): 
                if isValidNodeName(k): setattr(self, k, v)
        self.__path__ = path #<--- path will overwrite if present in members
        self.__ref__()

    def __ref__(self): self.__files__ = [ f for f in os.listdir(self.__path__) if os.path.isfile(os.path.join( self.__path__,f)) ]

    def __call__(self, file=None, relative=False): 
        p = os.path.join(self.__path__, file) if file else self.__path__
        if relative: p = os.path.relpath( p )
        return p

    def __repr__(self) -> str:return self.__str__()
    def __str__(self) -> str:  return f'Node @ [{self()}] :: {len([k for k in self.__dict__ if isValidNodeName(k)])} Dirs, {len(self.__files__)} Files'

    
class Store:
    r""" Abstracts a directory tree as collection of Nodes

    :param cwd_node: (bool) if True, adds ```os.getcwd()``` to Store
    :param auto_nodes: (list of str) adds nodes with their base-name as identifiers
    :param named_nodes: (dict of name:path) adds nodes with specified name as identifiers

    .. note:: Directories (nodes) with valid names are added as attributes. 
        Invalid directories and their sub-directories are ignored.
        A valid name should be an identifier and must not be dunder like i.e., does not start and end with `__` chars.
        
    """
    @staticmethod
    def build_tree_recursive(root_path):
        fL = os.listdir(root_path)
        node=Node(root_path)
        for f in fL:
            fp = os.path.join(root_path, f)
            if os.path.isdir(fp):
                k,v = __class__.build_tree_recursive(fp)
                setattr( node, k, v )
        return os.path.basename(root_path), node

    @staticmethod
    def call_tree_recursive(root, method):
        _ = getattr(root, method)() if isinstance(method, str) else method(root)
        for k,v in root.__dict__.items():
            if isValidNodeName(k): __class__.call_tree_recursive(v, method)
    
    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str: return f'Store: [{len(self.__roots__)}] : {self.__roots__.keys()}'

    def show(self):
        for k,v in self.__roots__.items():
            print(f'\n[KEY: {k}]')
            __class__.call_tree_recursive(v,print)

    def __init__(self, cwd_node:bool, *auto_nodes, **named_nodes) -> None:
        self.__roots__ = {}
        self.add_auto_nodes(*auto_nodes)
        self.add_named_nodes(**named_nodes)
        if cwd_node: self.add_node(os.getcwd())

    def refresh(self, node):
        if node in self.__roots__:
            res = self.add_node(self.__roots__[node].__path__)
            if not res: 
                del self.__roots__[node]
                delattr (self, node )

    def add_node(self, directory, create=False):
        r""" add a single node from a directory """
        if not os.path.isdir(directory):
            if not create: 
                print(f'[.] path @ {directory} is not a directory, skipping...')
                return False
    
        rp = os.path.abspath(directory)
        rn = os.path.basename(rp)
        if isValidNodeName(rn): 
            if create: os.makedirs(rp, exist_ok=True)
            n, r = __class__.build_tree_recursive(rp)
            if hasattr(self, n): print(f'[!] [{n}] already exsists in directories, will be overwritten!')
            setattr(self, n, r)
            self.__roots__[n]=r
            return True
        else:
            print(f'[!] [{rn}] is invalid name, skipping...')
            return False

    def add_named_nodes(self, **directories):
        r""" add multiple nodes from a dict  """
        for n,directory in directories.items():
            if not os.path.isdir(directory):
                print(f'[.] path @ {directory} is not a directory, skipping...')
                continue
            rp = os.path.abspath(directory)
            if isValidNodeName(n): 
                _, r = __class__.build_tree_recursive(rp)
                if hasattr(self, n): print(f'[!] [{n}] already exsists in directories, will be overwritten!')
                setattr(self, n, r)
                self.__roots__[n]=r
            else:
                print(f'[!] [{n}] is invalid name, skipping...')

    def add_auto_nodes(self, *directories):
        r""" add multiple nodes from a list  """
        for directory in directories:
            if not os.path.isdir(directory):
                print(f'[.] path @ {directory} is not a directory, skipping...')
                continue
            rp = os.path.abspath(directory)
            rn = os.path.basename(rp)
            if isValidNodeName(rn): 
                n, r = __class__.build_tree_recursive(rp)
                if hasattr(self, n): print(f'[!] [{n}] already exsists in directories, will be overwritten!')
                setattr(self, n, r)
                self.__roots__[n]=r
            else:
                print(f'[!] [{rn}] is invalid name, skipping...')


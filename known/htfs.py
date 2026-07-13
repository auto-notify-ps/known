
__doc__="""
# Usage


```python
from known.htfs import Requestor
endpoint = "127.0.0.1:8080"
```


### List actions

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action=""
```


### List (root) Folder Content

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="ListDir"
```

```python
Requestor.ListDir(endpoint, "")
```

### List (any) Folder Content

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="ListDir" --remote="store"
```

```python
Requestor.ListDir(endpoint, "store")
```

### Download File (as is)

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="GetFile" --remote="store/welcome.txt"
```

```python
Requestor.GetFile(endpoint, "store/welcome.txt")
```

### Download File (with custom name)

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="GetFile" --remote="store/welcome.txt" --local="welcome.log"
```

```python
Requestor.GetFile(endpoint, "store/welcome.txt", "welcome.log")
```

### Download Folder (as a zip file and extract it locally)

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="GetDir" --remote="store"
```

```python
Requestor.GetDir(endpoint, "store")
```

### Download Folder (as a zip file and extract it locally with custom name)

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="GetDir" --remote="" --local="base"
```

```python
Requestor.GetDir(endpoint, "", "base")
```

### Create Folder

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="PutDir" --remote="new"
```

```python
Requestor.PutDir(endpoint, "new")
```

### Upload Files to a Folder

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="PutFiles" --remote="new" --local "pubfs.py" "pubfs.service" "pubfs.sh"
```

```python
Requestor.PutFiles(endpoint, "new", "pubfs.py", "pubfs.service", "pubfs.sh")
```

### Delete File

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="Remove" --remote="new/pubfs.py"
```

```python
Requestor.Remove(endpoint, "new/pubfs.py")
```

### Delete Folder

```sh
python -m known.htfs --endpoint="127.0.0.1:8080" --action="Remove" --remote="new"
```

```python
Requestor.Remove(endpoint, "new")
```

### Help


```sh
python -m known.htfs --help
```

```python
help(Requestor)
```

"""


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def SetupLogging(logging, logfile, verbose, sep=" "):
    LOGFILE = None
    if logfile and verbose: 
        try: # Set up logging to a file # also output to the console
            LOGFILE = logfile
            format=f'%(asctime)s{sep}%(message)s'
            logging.basicConfig(filename=LOGFILE, level=logging.INFO, format=format)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(format)
            console_handler.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(console_handler)
        except: exit(f'[!] Logging could not be setup at {LOGFILE}')
    # ------------------------------------------------------------------------------------------
    if not verbose: sprint = lambda m: None
    else:
        if LOGFILE is None: sprint = lambda m: print(m)
        else: sprint = lambda m: logging.info(m) 
    # ------------------------------------------------------------------------------------------

    return sprint
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os, datetime, requests, shutil, zipfile
from urllib.parse import quote
from flask import Flask, request, abort, jsonify, send_file
from waitress import serve
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def TimeStamp(start='', sep='', end=''): 
    return (start + datetime.datetime.strftime(datetime.datetime.now(), 
    sep.join(["%Y", "%m", "%d", "%H", "%M", "%S", "%f"])) + end)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def ValidatePath(base, other):
    target = os.path.abspath(os.path.join(base, other))
    rel = os.path.relpath(target, base)
    if rel.startswith(os.pardir + os.sep) or rel == os.pardir: return None
    else: return target
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def Scan(path, exclude_hidden=False, include_size=False, include_extra=False):
    #if not os.path.exists(path): return []
    r""" Scans a directory using os.scandir call 
            returns a list of 6 or 9 tuple (name, path, isdir, isfile, islink, size, parent, fname, ext) """
    def SplitName(f):
        i = f.rfind('.')
        return (f, None) if i<0 else (f[0:i], f[i+1:])
    if exclude_hidden:  
        if include_size: 
            if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size, os.path.dirname(os.path.abspath(x.path)), *SplitName(x.name)) for x in os.scandir(path) if not x.name.startswith(".")]
            else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size) for x in os.scandir(path) if not x.name.startswith(".")]

        else:            
            if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              , os.path.dirname(os.path.abspath(x.path)), *SplitName(x.name)) for x in os.scandir(path) if not x.name.startswith(".")]
            else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              ) for x in os.scandir(path) if not x.name.startswith(".")]
    else:
        if include_size: 
            if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size, os.path.dirname(os.path.abspath(x.path)), *SplitName(x.name)) for x in os.scandir(path)]
            else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size) for x in os.scandir(path)]
        else:
            if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              ,os.path.dirname(os.path.abspath(x.path)), *SplitName(x.name)) for x in os.scandir(path)]
            else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              ) for x in os.scandir(path)]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def ReScan(path, exclude_hidden=False, include_size=False, include_extra=False):
    r""" Recursively Scans a directory using os.scandir """
    res = []
    pending = [path]
    while pending:
        try:
            ls = Scan(pending.pop(0), exclude_hidden=exclude_hidden, include_size=include_size, include_extra=include_extra)
            for l in ls: # name, path, isdir, isfile, islink, size, parent, (fname, ext)
                res.append(l)
                if l[2]: (pending.append(l[1]) if not l[4] else None)
        except: pass
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def DirectoryListing(data, dirpath, recurse, hidden): return {name:dict(path=os.path.relpath(path, data), size=size, itype=( ("📁" if isdir else "") + ("🗒️" if isfile else "") + ("🔗" if islink else "") )) for name, path, isdir, isfile, islink, size in (ReScan if recurse else Scan)(dirpath, exclude_hidden=(not hidden), include_size=True, include_extra=False)}
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def Zipped(src_path, zip_path):
    src_path = os.path.abspath(src_path)
    parent_dir = os.path.dirname(src_path)
    #base_name = os.path.basename(src_path)
    #zip_path = os.path.join(parent_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_path):
            # preserve empty directories too
            rel_root = os.path.relpath(root, parent_dir)
            zf.write(root, rel_root)
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, parent_dir)
                zf.write(full_path, rel_path)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def UnZipped(zip_path, target_path):
    with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(target_path)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class HTFS:

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    ALL_HOST = "0.0.0.0"
    LOCAL_HOST = "127.0.0.1"
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(self, name, base, data, printer, 
                source:str, host:str, port:str, threads:int, connection_limit:int, max_request_body_size:str,
                recurse_list:bool=False, hidden_list:bool=False, allow_del:bool=False, allow_put:bool=False, allow_post:bool=False, allow_get:bool=True):
        
        if not base: base = os.getcwd()
        base=os.path.abspath(base)
        os.makedirs(base, exist_ok=True)
        self.base = base

        if not data: data = os.getcwd()
        data=os.path.abspath(data)
        os.makedirs(data, exist_ok=True)
        self.data = data

        self.app = Flask(name, 
                instance_relative_config = True, 
                static_folder=self.base, 
                template_folder=self.base, 
                instance_path = self.base,)
        self.printer, self.host, self.port, self.threads, self.connection_limit, self.max_request_body_size = \
            printer, host, port, threads, connection_limit, max_request_body_size
        self.source = source if source else f'{(self.LOCAL_HOST if self.host == self.ALL_HOST else self.host)}:{self.port}' 
        self.recurse_list=recurse_list
        self.hidden_list=hidden_list
        self.allow_del, self.allow_put, self.allow_post, self.allow_get = allow_del, allow_put, allow_post, allow_get
        self.register()

    def __call__(self): 
        self.printer(f'[START] 🟢 Access via {self.source}')
        # ===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++
        def str2bytes(size):
            if isinstance(size, str):
                sizes = dict(BB=2**0, KB=2**10, MB=2**20, GB=2**30, TB=2**40)
                return int(float(size[:-2])*sizes.get(size[-2:].upper(), 0))
            else: return max(1024, int(abs(size)))
        # ===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++
        start_time = datetime.datetime.now()
        # ===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++
        serve(self.app, 
            host = self.host, 
            port = self.port, 
            url_scheme = 'http', 
            threads = self.threads, 
            connection_limit = self.connection_limit,
            max_request_body_size = str2bytes(self.max_request_body_size),
            ) # https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html
        # ===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++
        end_time = datetime.datetime.now()
        # ===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++===+++
        self.printer('[END] 🛑 Stopped, uptime was {}'.format(end_time - start_time))
        return 

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def ls(self, dirpath):  return DirectoryListing(self.data, dirpath, self.recurse_list, self.hidden_list)

    def register(self):
        methods = []
        if self.allow_get: methods.append("GET")
        if self.allow_post: methods.append("POST")
        if self.allow_put: methods.append("PUT")
        if self.allow_del: methods.append("DELETE")
        self.methods = set(methods)

        self.app.add_url_rule("/",                  view_func=self.route,  methods=methods, defaults={"req_path": ""})
        self.app.add_url_rule("/<path:req_path>",   view_func=self.route,  methods=methods,                          )
    
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def route(self, req_path):
        if request.method not in self.methods: return abort(405)

        #-------------------------------------
        #PATH BEHAVIOR
        #-------------------------------------
    
        abs_path = ValidatePath(self.data, req_path)
        self.printer(f'[{request.method}] ({req_path})::{abs_path}')
        if abs_path is None: return abort(404)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if request.method=='GET': # download files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if os.path.isfile(abs_path): # send the file
                self.printer(f'[GET] ⚡ Downloading file {req_path} via {request.remote_addr}')
                return send_file(abs_path, as_attachment=("?" in request.args)) 
            elif os.path.isdir(abs_path): # send directory listing
                if "?" in request.args:
                    dirname = os.path.basename(abs_path)
                    zip_path = os.path.join(self.base, TimeStamp(start=f"{dirname}_", end=".zip"))
                    Zipped(abs_path, zip_path)
                    return send_file(zip_path, as_attachment=True) 
                else: return jsonify(self.ls(abs_path))
            else: return abort(404)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif request.method=='POST': 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if os.path.isdir(abs_path): # upload files to existing folders
                success=False
                for fk, fv in request.files.items(): 
                    try: 
                        fv.save(os.path.join(abs_path, fk))
                        self.printer(f'[POST] ✅ Upload to dir {req_path, fk} via {request.remote_addr}')
                    except: self.printer(f'[POST] ❌ Upload to dir {req_path, fk} via {request.remote_addr}')
                success=True
                return f'{success}', (200 if success else 500)
            else: return abort(404) # upload file to exact path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif request.method=='PUT': # create dir
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            success=False
            try: 
                os.makedirs(abs_path, exist_ok=True)
                success=True
                self.printer(f'[PUT] ✅ Create dir {req_path} via {request.remote_addr}')
            except: self.printer(f'[PUT] ❌ Create dir {req_path} via {request.remote_addr}')
            return f'{success}', (200 if success else 500)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif request.method=='DELETE': # delete file or folders
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not os.path.exists(abs_path): return f"Does not exist", 200
            if os.path.isdir(abs_path):
                success=False
                try: 
                    shutil.rmtree(abs_path)
                    success=True
                    self.printer(f'[DELETE] ✅ Remove dir {req_path} via {request.remote_addr}')
                except: self.printer(f'[DELETE] ❌ Remove dir {req_path} via {request.remote_addr}')
                return f'{success}', (200 if success else 500)
            else:
                success=False
                try:
                    os.remove(abs_path)
                    success=True
                    self.printer(f'[DELETE] ✅ Remove file {req_path} via {request.remote_addr}')
                except: self.printer(f'[DELETE] ❌ Remove file {req_path} via {request.remote_addr}')
                return f'{success}', (200 if success else 500)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else: return abort(405)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Requestor:

    # #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # @staticmethod
    # def Index(endpoint, **kwargs):
    #     argstr = "?" if kwargs else ""
    #     for k,v in kwargs.items(): argstr+=f'{k}={v}&'
    #     try:
    #         response = requests.get(f"{endpoint}{argstr}")
    #         success = (response.status_code == 200)
    #         if success: result = response.json() if kwargs else response.text
    #         else: result = None
    #     except: success, result = False, None
    #     return success, result
    # #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def GetFile(endpoint, remote_path, local_path=None, chunk_size=None):
        if not local_path: local_path=os.path.basename(remote_path)
        do_stream = bool(chunk_size)
        if do_stream: chunk_size = int(chunk_size)
        try:
            response = requests.get(f"{endpoint}/{quote(remote_path, safe='/')}", stream=do_stream)
            success = (response.status_code == 200)
            if success:
                with open(local_path, "wb") as f:
                    if do_stream:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk: f.write(chunk)
                    else: f.write(response.content)
        except: success = False
        return success
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def GetDir(endpoint, remote_path, local_path=None, chunk_size=None):
        if not local_path: local_path=os.path.abspath(".")
        zip_path = os.path.join(local_path, TimeStamp(start=f'{os.path.basename(remote_path)}', end='.zip'))
        do_stream = bool(chunk_size)
        if do_stream: chunk_size = int(chunk_size)
        try:
            response = requests.get(f"{endpoint}/{quote(remote_path, safe='/')}??", stream=do_stream)
            success = (response.status_code == 200)
            if success:
                with open(zip_path, "wb") as f:
                    if do_stream:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk: f.write(chunk)
                    else: f.write(response.content)
                UnZipped(zip_path, local_path)
                os.remove(zip_path)
        except: success = False
        return success
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def ListDir(endpoint, remote_path):
        #try:
        response = requests.get(f"{endpoint}/{quote(remote_path, safe='/')}")
        success = (response.status_code == 200)
        result = response.json()
        #except: success, result = False, None
        return success, result
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def PutFiles(endpoint, remote_path, *local_paths):
        files = {}
        try:
            for local_file in local_paths: files[os.path.basename(local_file)] = open(local_file, "rb") 
            response = requests.post(f"{endpoint}/{quote(remote_path, safe='/')}", files=files)
            success = (response.status_code == 200)
        except: success = False
        finally:
            for handle in files.values(): handle.close()
        return success
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def PutDir(endpoint, remote_path):
        try:
            response = requests.put(f"{endpoint}/{quote(remote_path, safe='/')}")
            success = (response.status_code == 200)
        except: success = False
        return success
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @staticmethod
    def Remove(endpoint, remote_path):
        try:
            response = requests.delete(f"{endpoint}/{quote(remote_path, safe='/')}")
            success = (response.status_code == 200)
        except: success = False
        return success
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=





if __name__=='__main__':

    # ------------------------------------------------------------------------------------------
    import argparse
    # ------------------------------------------------------------------------------------------
    argp = argparse.ArgumentParser()
    argp.add_argument('--endpoint',        type=str, default='',                help='calls requestor, with this endpoint')
    argp.add_argument('--action',          type=str, default='',                help='calls requestor, with this action')
    argp.add_argument('--remote',          type=str, default='',                 help='calls requestor, with this remote_path')
    argp.add_argument('--local',           nargs='*', type=str,                  help='calls requestor, with this local_path')
    # ------------------------------------------------------------------------------------------
    argp.add_argument('--base',             type=str, default='base',                help='base directory for app')
    argp.add_argument('--data',             type=str, default='data',                help='data directory for app')
    argp.add_argument('--host',             type=str, default='127.0.0.1',       help='waitress server-interface, keep 0.0.0.0 to serve on all interfaces')
    argp.add_argument('--port',             type=str, default='8080',            help='waitress server-port')
    argp.add_argument('--source',           type=str, default='',                help='enpoint for file transfer - leave blank for same as host port')
    argp.add_argument('--maxupsize',        type=str, default='1MB',             help='http_body_size for waitress') 
    argp.add_argument('--maxconnect',       type=int, default=100,               help='maximum number of connections allowed') 
    argp.add_argument('--threads',          type=int, default=2,                 help='waitress thread count') 
    argp.add_argument('--log',              type=str, default='',                help='specify a file or keep blank for no logging')
    argp.add_argument('--verbose',          type=int, default=1,                 help='set to 0 for no verbose')
    # ------------------------------------------------------------------------------------------
    argp.add_argument('--list_recursive', type=int, default=0,                 help='set to 1 for recursive list')
    argp.add_argument('--list_hidden',    type=int, default=0,                 help='set to 1 for hidden file list')
    argp.add_argument('--allow_get',      type=int, default=1,                 help='')
    argp.add_argument('--allow_post',     type=int, default=0,                 help='')
    argp.add_argument('--allow_put',      type=int, default=0,                 help='')
    argp.add_argument('--allow_del',      type=int, default=0,                 help='')

    
    # ------------------------------------------------------------------------------------------
    parsed = argp.parse_args()
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    if parsed.endpoint:
        if not parsed.action: 
            print(f'Provide an action from {[f for f in Requestor.__dict__.keys() if not f.startswith("__")]}')
            exit(f'No action provided')
        ActionF = getattr(Requestor, parsed.action)
        LocalF = [] if not parsed.local else parsed.local
        RemoteF = parsed.remote if parsed.remote else "" 
        print(f"{ActionF.__name__}({parsed.endpoint}, {RemoteF}, {LocalF})")
        Result = ActionF(parsed.endpoint, RemoteF, *LocalF)
        print(Result)
    else:
        import logging
        HTFS(
            name=__name__,
            base=parsed.base,
            data=parsed.data,
            printer=SetupLogging(logging, parsed.log, parsed.verbose),
            source=parsed.source,
            host=parsed.host,
            port=parsed.port, 
            threads=parsed.threads, 
            connection_limit=parsed.maxconnect, 
            max_request_body_size=parsed.maxupsize,
            recurse_list=bool(parsed.list_recursive),
            hidden_list=bool(parsed.list_hidden),
            allow_del=bool(parsed.allow_del),
            allow_put=bool(parsed.allow_put),
            allow_post=bool(parsed.allow_post),
            allow_get=bool(parsed.allow_get),
        )()
    # ------------------------------------------------------------------------------------------

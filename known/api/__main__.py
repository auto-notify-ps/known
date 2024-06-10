__doc__="""
HTTP API Request-Response model
"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------
# imports 
#-----------------------------------------------------------------------------------------
import os, argparse, datetime, importlib, importlib.util
from enum import IntEnum
#PYDIR = os.path.dirname(__file__) # script directory of __main__.py
try:

    from flask import Flask, request, send_file, abort
    from waitress import serve
    from http import HTTPStatus
    from shutil import rmtree
except: exit(f'[!] The required packages missing:⇒ pip install Flask waitress')
#-----------------------------------------------------------------------------------------



# ==============================================================================================================
# Common Functions 
# NOTE: common functions are repeated in all modular servers so that they can act as stand alone
# ==============================================================================================================


class HRsizes: # human readable size like  000.000?B
    mapper = dict(KB=2**10, MB=2**20, GB=2**30, TB=2**40)
    def tobytes(size): return int(float(size[:-2])*__class__.mapper.get(size[-2:].upper(), 0))

class EveryThing: # use as a set that contains everything (use the 'in' keyword)
    def __contains__(self, x): return True

def ImportCustomModule(python_file:str, python_object:str, do_initialize:bool):
    r""" Import a custom module from a python file and optionally initialize it """
    cpath = os.path.abspath(python_file)
    failed=""
    if os.path.isfile(cpath): 
        try: 
            # from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
            cspec = importlib.util.spec_from_file_location("", cpath)
            cmodule = importlib.util.module_from_spec(cspec)
            cspec.loader.exec_module(cmodule)
            success=True
        except: success=False #exit(f'[!] Could import user-module "{module_path}"')
        if success: 
            if python_object:
                try:
                    cmodule = getattr(cmodule, python_object)
                    if do_initialize:  cmodule = cmodule()
                except:         cmodule, failed = None, f'[!] Could not import object {python_object} from module "{module_path}"'
        else:                   cmodule, failed = None, f'[!] Could not import module "{module_path}"'
    else:                       cmodule, failed = None, f"[!] File Not found @ {cpath}"
    return cmodule, failed

# def show(x, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> str:
#     res = ""
#     for d in dir(x):
#         if not (d.startswith(sw) or d.endswith(ew)):
#             v = ""
#             try:
#                 v = getattr(x, d)
#             except:
#                 v='?'
#             res+=f'(👉{d}👈 {cep} 🔹{v}🔸{sep}'
#     return res
    
# ==============================================================================================================


#-----------------------------------------------------------------------------------------
# Parse arguments 
#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
#-----------------------------------------------------------------------------------------
# user module related
# s/d/f/s.py:sd43_:ioi:klj:m:
parser.add_argument('--user', type=str, default='', help="path of python script")
parser.add_argument('--object', type=str, default='', help="the python object inside python script")
parser.add_argument('--callable', type=int, default=0, help="if true, calls the python object (to initialize) - works with object only")
parser.add_argument('--handle', type=str, default='handle', help="name of the function that handles the api calls")

# server hosting related
parser.add_argument('--host', type=str, default='0.0.0.0', help="IP-Addresses of interfaces to start the server on, keep default for all interfaces")
parser.add_argument('--port', type=str, default='8080', help="server's port")
parser.add_argument('--maxH', type=str, default='100MB', help="max_request_body_size in the HTTP request")
parser.add_argument('--maxB', type=str, default='100MB', help="max_request_body_size in the HTTP request")
parser.add_argument('--limit', type=int, default=10, help="maximum number of connections allowed with the server")
parser.add_argument('--threads', type=int, default=1, help="maximum number of threads used by server")
parser.add_argument('--allow', type=str, default='', help="the remote address that match this CSV list will be allowed to access API, keep blank to allow all")
parser.add_argument('--storage', type=str, default='', help="the path on server where files will be sent for download")
#-----------------------------------------------------------------------------------------
parsed = parser.parse_args()
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
# Import and Initialize user module (handler)
def default_handle(request_content, request_type, request_tag): return f"HANDLER\n{request_type=}\t{request_tag=}\n{request_content=}", "MESG", "Handle-Tag"
#-----------------------------------------------------------------------------------------
user_handle = default_handle # ---> global variable
if parsed.user:
    user_module, reason = ImportCustomModule(os.path.abspath(parsed.user), parsed.object, bool(parsed.callable))
    if user_module is None:                         exit(f'[!] FAILED importing user module :: {reason}')
    if not hasattr(user_module, parsed.handle):     exit(f'[!] Handler Method not found "{parsed.handle}"')
    user_handle = getattr(user_module, parsed.handle)
    
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# application setting and instance
# ------------------------------------------------------------------------------------------
app = Flask(__name__)
if parsed.allow:
    allowed = set(parsed.allow.split(','))
    if '' in allowed: allowed.remove('')
else: allowed = EveryThing()
if parsed.storage:
    storage_path = os.path.abspath(parsed.storage)
    os.makedirs(storage_path, exist_ok=True)
else: storage_path = os.getcwd()

app.config['allow'] =      allowed
app.config['storage'] =    storage_path
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
# NOTE on return type
# ... type must be a string, dict, list, 
# ... type can be this but ignored: tuple with headers or status, Response instance, or WSGI callable
#-----------------------------------------------------------------------------------------
@app.route('/', methods =['GET', 'POST'])
def home():
    global user_handle
    
    if request.method == 'POST':
        request_from = request.environ['REMOTE_HOST'] 
        if request_from in app.config['allow']:
            xtype, xtag = request.headers.get('User-Agent'), request.headers.get('Warning')
            
            if   xtype=='MESG': xcontent = request.get_data().decode('utf-8')
            elif xtype=='BYTE': xcontent = request.get_data()
            elif xtype=='FORM': xcontent = request.form, request.files
            elif xtype=='JSON': xcontent = request.get_json()
            else:               xcontent = None               
            
            if xcontent is None: return_object, return_code, return_headers = f'Type "{xtype}" is not a valid content type', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                return_object, return_hint, return_tag = user_handle(xcontent, xtype, xtag)
                return_headers = {'User-Agent':f"{return_hint}", 'Warning':f"{return_tag}"}
                if isinstance(return_object, (str, dict, list, bytes)): return_code = HTTPStatus.OK
                else: return_object, return_code, return_headers = f"[!] Invalid Return type from handler [{type(return_object)}]", HTTPStatus.NOT_FOUND, {}
        else: return_object, return_code, return_headers = f"[!] You are not allowed to POST", HTTPStatus.NOT_ACCEPTABLE, {}

    elif request.method == 'GET':     
        return_object = f'<pre>[Known.api]@{__file__}\n'
        for k,v in parsed._get_kwargs(): return_object+=f'\n\t{k}\t{v}\n'
        return_object+='</pre>'
        return_code = HTTPStatus.OK
        return_headers={}

    else: return_object, return_code, return_headers = f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST, {}
    # only use either one of data or json in post request (not both)
    return return_object, return_code, return_headers


    
@app.route('/store', methods =['GET'])
def storagelist(): # an overview of all storage paths and the files in them
    basedir = app.config['storage']
    return {os.path.relpath(root, basedir) : files for root, directories, files in os.walk(basedir)}, HTTPStatus.OK, {'User-Agent': f'{basedir}', 'Warning': 'LIST'}

@app.route('/store/', methods =['GET'])
def storageroot(): # an overview of all storage paths and the files in them
    rw, dw, fw = next(iter(os.walk(app.config['storage'])))
    rel_path = os.path.relpath(rw, app.config['storage'])
    return dict(base=os.path.relpath(rw, app.config['storage']), folders=dw, files=fw), HTTPStatus.OK, {'User-Agent': f'{rel_path}', 'Warning': 'DIR'}


@app.route('/store/<path:req_path>', methods =['GET', 'POST', 'PUT', 'DELETE'])
def storage(req_path): # creates a FileNotFoundError
    abs_path = os.path.join(app.config['storage'], req_path) # Joining the base and the requested path
    rel_path = os.path.relpath(abs_path, app.config['storage'])
    if request.method=='GET': # trying to download that file or view a directory
        if os.path.exists(abs_path):
            if os.path.isdir(abs_path):     
                rw, dw, fw = next(iter(os.walk(abs_path)))
                return dict(base=rel_path, folders=dw, files=fw), HTTPStatus.OK, {'User-Agent': f'{rel_path}', 'Warning': 'DIR'}
            else: 
                resx = send_file(abs_path) 
                resx.headers['User-Agent'] = os.path.basename(abs_path) # 'asve_as'
                resx.headers['Warning'] = 'FILE'
                return resx
        else: return f'Path not found: {abs_path}', HTTPStatus.NOT_FOUND
    elif request.method=='POST': # trying to create new file or replace existing file
        if os.path.isdir(abs_path):
            return f'A directory already exists at {abs_path} - cannot create a file there', HTTPStatus.NOT_ACCEPTABLE
        else:
            try: 
                with open(abs_path, 'wb') as f: f.write(request.get_data())
                return      f"Created File @ {abs_path}", HTTPStatus.OK, {'User-Agent': f'{rel_path}', 'Warning': 'FILE'}
            except: return  f"File cannot be created @ {abs_path}", HTTPStatus.NOT_ACCEPTABLE
    elif request.method=='PUT': # trying to create new directory
        if os.path.isfile(abs_path):
            return f'A file already exists at {abs_path} - cannot create a filder there', HTTPStatus.NOT_ACCEPTABLE
        else:
            os.makedirs(abs_path, exist_ok=True)
            return f"Created Folder @ {abs_path}", HTTPStatus.OK, {'User-Agent': f'{rel_path}', 'Warning': 'DIR'}
    elif request.method=='DELETE': # trying to delete a file or folder
        if os.path.isfile(abs_path):
            os.remove(abs_path)
            return f"Deleted File @ {abs_path}", HTTPStatus.OK, {'User-Agent': f'{rel_path}', 'Warning': 'FILE'}
        elif os.path.isdir(abs_path):
            rok = True
            if int(request.headers.get('User-Agent')):
                try: rmtree(abs_path)
                except: rok=False
            else:
                try: os.rmdir(abs_path)
                except: rok=False
            if rok: return  f"Folder deleted @ {abs_path}", HTTPStatus.OK, {'User-Agent': f'{rel_path}', 'Warning': 'DIR'}
            else:   return  f'Cannot delete folder at {abs_path}', HTTPStatus.NOT_ACCEPTABLE
        else: return f'Cannot delete at {abs_path} - not a file or folder', HTTPStatus.NOT_ACCEPTABLE
    else: return f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST

#%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
start_time = datetime.datetime.now()
print('◉ start server @ [{}]'.format(start_time))
serve(app, # https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html
    host = parsed.host,          
    port = parsed.port,          
    url_scheme = 'http',     
    threads = parsed.threads,    
    connection_limit = parsed.limit,
    max_request_header_size = HRsizes.tobytes(parsed.maxH),
    max_request_body_size = HRsizes.tobytes(parsed.maxB),
    
)
#<-------------------DO NOT WRITE ANY CODE AFTER THIS
end_time = datetime.datetime.now()
print('')
print('◉ stop server @ [{}]'.format(end_time))
print('◉ server up-time was [{}]'.format(end_time - start_time))

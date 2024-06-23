__doc__="""
HTTP API Request-Response model.
This module is designed to enable communication between a remote python process and a central server using HTTP API.
"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] Can not import {__name__}:{__file__}')
#-----------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------
# imports 
#-----------------------------------------------------------------------------------------
import os, argparse, datetime, importlib, sys
from .client import HeaderType, RequestContentType, ResponseContentType, StoreType
#PYDIR = os.path.dirname(__file__) # script directory of __main__.py
try:

    from flask import Flask, request, send_file, abort
    from waitress import serve
    from http import HTTPStatus
    from shutil import rmtree
except: exit(f'[!] Required packages missing')
#-----------------------------------------------------------------------------------------



# ==============================================================================================================
# Common Functions 
# NOTE: common functions are repeated in all modular servers so that they can act as stand alone
# ==============================================================================================================

class Fake:
    def __len__(self): return len(self.__dict__)
    def __init__(self, **kwargs):
        for k,v in kwargs.items():setattr(self, k, v)
    #def _get_kwargs(self): return self.__dict__.items()

class HRsizes: # human readable size like  000.000?B
    mapper = dict(KB=2**10, MB=2**20, GB=2**30, TB=2**40)
    def tobytes(size): return int(float(size[:-2])*__class__.mapper.get(size[-2:].upper(), 0))

class EveryThing: # use as a set that contains everything (use the 'in' keyword)
    def __contains__(self, x): return True

# ==============================================================================================================
# this is useful for docker
# we try to read args from os env variables
def DEFAULT_CONFIG_GENERATE(env): return '\nconfig = {\n' + f"""
    'maxH'        : "{env.get('maxH','0.25GB')}", # maximum http header size
    'maxB'        : "{env.get('maxB', '1.0GB')}", # maximum http body size
    'limit'       : {int(env.get('limit', 5))}, # maximum connection limit to the server
    'port'        : "{env.get('port', '8080')}", # server port
    'host'        : "{env.get('host', '0.0.0.0')}", # server address (keep 0.0.0.0 to run on all IPs)
    'allow'       : "{env.get('allow', '')}", # a comma-seperated list of host IP address that are allowed to POST (keep blank to allow all)
    'uids'       : "{env.get('uids', '')}", # a comma-seperated list of client uids that are allowed to POST (keep blank to allow all)
    'threads'     : {int(env.get('threads', 1))}, # no of threads on the server
    'storage'     : "{env.get('storage', '')}", # the path of storage folder (keep blank to use `os.getcwd()` or set as `None` to not use storage)
""" +'\n}'+ """
def handle(request_content:object, request_type:str) -> (object, str, str):
    # handle an incoming request from client 
    # NOTE: only handles `send_` type requests and not `path_` type requests

    # use arguments `request_type` and `request_content` to access the client request data

    # process the data ...
    print(request_type, type(request_content))
    
    # return a valid response containing (response_content, response_type, response_tag)
    response_content = "default_response"       # response_content should be one of (str, dict, list, bytes)
    response_type = "MESG"                      # response_type should be [MESG, BYTE, JSON]
    response_tag = "default_handle"             # response_tag is just a string
    return response_content, response_type, response_tag

"""

def DEFAULT_CONFIG_WRITE(file_path, env):
    with open(file_path, 'w', encoding='utf-8') as f: f.write(DEFAULT_CONFIG_GENERATE(env))

#-----------------------------------------------------------------------------------------
# Parse arguments 
#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
#for k,v in ALL_ARGS.items(): parser.add_argument(f'--{k}', **v)
parser.add_argument('--dir', type=str, default='', help="path of workspace directory")
parser.add_argument('--config', type=str, default='', help="path of config-file")
parsed = parser.parse_args()
#-----------------------------------------------------------------------------------------

WORKDIR = f'{parsed.dir}'                               # define working dir - contains all bases
if not WORKDIR: WORKDIR = os.getcwd()    # if still not specified, set as default
print(f'↪ Workspace directory is {WORKDIR}')
try: os.makedirs(WORKDIR, exist_ok=True)
except: exit(f'[!] Workspace directory was not found and could not be created')
sys.path.append(WORKDIR)  # otherwise append to sys.path

#-----------------------------------------------------------------------------------------
# ==> read configurations
#-----------------------------------------------------------------------------------------
CONFIG, HANDLE = 'config', 'handle'
CONFIG_MODULE = parsed.config if parsed.config else 'api' # the name of configs module
CONFIGS_FILE = f'{CONFIG_MODULE}.py' # the name of configs file


# check if 'configs.py` exsists or not`
CONFIGS_FILE_PATH = os.path.join(WORKDIR, CONFIGS_FILE) # should exsist under workdir
first_exit=False
if not os.path.isfile(CONFIGS_FILE_PATH):
    print(f'↪ Creating default config "{CONFIGS_FILE}" from environment...')
    DEFAULT_CONFIG_WRITE(CONFIGS_FILE_PATH, dict(os.environ))
    first_exit=True
try: 
    c_module = importlib.import_module(f'{CONFIG_MODULE}')
    print(f'↪ Imported config-module "{CONFIG_MODULE}" from {c_module.__file__}')
except: exit(f'[!] Could import configs module "{CONFIG_MODULE}" at "{CONFIGS_FILE_PATH[:-3]}"')
try:
    print(f'↪ Reading config from {CONFIG_MODULE}.{CONFIG}')
    config_dict = getattr(c_module, CONFIG)
    print(f'  ↦ type:{type(config_dict)}')
except:
    exit(f'[!] Could not read config from {CONFIG_MODULE}.{CONFIG}')

if not isinstance(config_dict, dict): 
    try: config_dict=config_dict()
    except: pass
if not isinstance(config_dict, dict): raise exit(f'Expecting a dict object for config')

try: 
    print(f'↪ Building config from {CONFIG_MODULE}.{CONFIG}')
    #for k,v in config_dict.items(): print(f'  ↦ {k}\t\t{v}')
    args = Fake(**config_dict)
except: exit(f'[!] Could not read config')
if not len(args): exit(f'[!] Empty or Invalid config provided')
try:
    print(f'↪ Getting handle from {CONFIG_MODULE}.{HANDLE}')
    user_handle = getattr(c_module, HANDLE)
    print(f'  ↦ type:{type(user_handle)}')
except:
    exit(f'[!] Could not get handle from {CONFIG_MODULE}.{HANDLE}')

if first_exit:
    print(f'↪ Configuration was built! Press [enter] to start server now ↦↦↦↦')
    if input(): exit('Server was not started!')
# ------------------------------------------------------------------------------------------
# application setting and instance
# ------------------------------------------------------------------------------------------
app = Flask(__name__)
if args.allow:
    allowed = set(args.allow.split(','))
    if '' in allowed: allowed.remove('')
else: allowed = EveryThing()
if args.storage:
    storage_path = os.path.abspath(args.storage)
    os.makedirs(storage_path, exist_ok=True)
else: 
    storage_path = None if args.storage is None else os.path.abspath(WORKDIR) #os.getcwd()

if args.uids:
    uid_allow = set(args.uids.split(','))
    if '' in uid_allow: uid_allow.remove('')
else: uid_allow = EveryThing()

app.config['allow'] =      allowed
app.config['storage'] =    storage_path
app.config['uids'] =       uid_allow
# tag is assigned to a uid after it makes a login request - tag is used by client in future requests
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

            # The clients making post request will have to provide these two headers
            # headers are only read for post requests from allowed users
            xtag, xtype = request.headers.get(HeaderType.XTAG), request.headers.get(HeaderType.XTYPE)
            if xtag not in app.config['uids']: xcontent = None
            elif xtype is None:             xcontent = None
            #------------------------------------------------------------------------------- Read from the reuest made by client
            elif xtype==RequestContentType.MESG: xcontent = request.get_data().decode('utf-8')
            elif xtype==RequestContentType.BYTE: xcontent = request.get_data()
            elif xtype==RequestContentType.FORM: xcontent = request.form, request.files
            elif xtype==RequestContentType.JSON: xcontent = request.get_json()
            #-------------------------------------------------------------------------------
            else:                         xcontent = None               
            
            if xcontent is not None:
                return_object, return_type, return_tag = user_handle(xcontent, xtype)
                if isinstance(return_object, (str, dict, list, bytes)) and (return_type in ResponseContentType.ALL): 
                    return_code = HTTPStatus.OK
                    return_headers = {HeaderType.XTAG :return_tag, HeaderType.XTYPE:return_type} #<-- headers are only sent when content and types are valid
                else:   return_object, return_code, return_headers = f"[!] Invalid response from handler [{type(return_object)}::{return_type}:{return_tag}]", HTTPStatus.NOT_FOUND, {}
            else:       return_object, return_code, return_headers = f'[!] Type "{xtype}" is not a valid content type', HTTPStatus.NOT_ACCEPTABLE, {}
        else:           return_object, return_code, return_headers = f"[!] You are not allowed to POST", HTTPStatus.NOT_ACCEPTABLE, {}
    elif request.method == 'GET':     
        return_object = f'<pre>[Known.api]@{__file__}\n[Workdir]@{WORKDIR}\n[Storage]@{app.config["storage"]}\n[Config]@{CONFIGS_FILE_PATH}</pre>'
        #for k,v in args._get_kwargs(): return_object+=f'\n\t{k}\t{v}\n'
        return_code, return_headers = HTTPStatus.OK, {}
    else: return_object, return_code, return_headers = f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST, {}
    
    return return_object, return_code, return_headers


# Storage urls for file-storage api
# tag specifies a name of file, type specifies if its a overall-view, a directory listing or a file


@app.route('/store', methods =['GET'])
def storageview(): # an overview of all storage paths and the files in them
    uid = request.headers.get(HeaderType.XTAG)
    if uid not in app.config['uids']:  return f'Invalid client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
    basedir = app.config['storage']
    if basedir is None: return_object, return_code, return_headers = {}, HTTPStatus.NOT_FOUND, {}
    else: return_object, return_code, return_headers =  {os.path.relpath(root, basedir) : files for root, directories, files in os.walk(basedir)}, HTTPStatus.OK, {HeaderType.XTAG: f'{basedir}', HeaderType.XTYPE: StoreType.HOME}
    return return_object, return_code, return_headers

@app.route('/store/', methods =['GET'])
def storageroot(): # root dir
    uid = request.headers.get(HeaderType.XTAG)
    if uid not in app.config['uids']:  return f'Invalid client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
    basedir = app.config['storage']
    if basedir is None: return_object, return_code, return_headers = {}, HTTPStatus.NOT_FOUND, {}
    else:
        rw, dw, fw = next(iter(os.walk(basedir)))
        fs = [os.path.getsize(os.path.join(rw, fwi)) for fwi in fw]
        rel_path = os.path.relpath(rw, basedir)
        return_object = dict(base=os.path.relpath(rw, basedir), folders=dw, files={k:round(v/1024,2) for k,v in zip(fw,fs)}) # size in KB
        return_code = HTTPStatus.OK
        return_headers = {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.DIR}
    return return_object, return_code, return_headers

@app.route('/store/<path:req_path>', methods =['GET', 'POST', 'PUT', 'DELETE'])
def storage(req_path): # creates a FileNotFoundError
    basedir = app.config['storage']
    if basedir is None: return_object, return_code, return_headers = {}, HTTPStatus.NOT_FOUND, {}
    else:
        abs_path = os.path.join(basedir, req_path) # Joining the base and the requested path
        rel_path = os.path.relpath(abs_path, basedir)

        uid = request.headers.get(HeaderType.XTAG)
        if uid not in app.config['uids']:  return f'Invalid client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}

        if request.method=='GET': # trying to download that file or view a directory
            if os.path.exists(abs_path):
                if os.path.isdir(abs_path):     
                    rw, dw, fw = next(iter(os.walk(abs_path)))
                    fs = [os.path.getsize(os.path.join(rw, fwi)) for fwi in fw]
                    return_object = dict(base=rel_path, folders=dw, files={k:round(v/1024,2) for k,v in zip(fw,fs)}) # size in KB
                    return_code = HTTPStatus.OK
                    return_headers = {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.DIR}
                else: 
                    resx = send_file(abs_path) 
                    resx.headers[HeaderType.XTAG] = os.path.basename(abs_path) # 'save_as'
                    resx.headers[HeaderType.XTYPE] = StoreType.FILE
                    return resx #<-----RETURNING HERE
            else: return_object, return_code, return_headers = f'Path not found: {abs_path}', HTTPStatus.NOT_FOUND, {}


        elif request.method=='POST': # trying to create new file or replace existing file
            if os.path.isdir(abs_path):
                return_object, return_code, return_headers = f'Cannot create file # {abs_path} - folder already exists', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                try: 
                    with open(abs_path, 'wb') as f: f.write(request.get_data())
                    return_object, return_code, return_headers =  f"File created @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                except: return_object, return_code, return_headers =   f"Cannot create file @ {abs_path}", HTTPStatus.NOT_ACCEPTABLE, {}


        elif request.method=='PUT': # trying to create new directory
            if os.path.isfile(abs_path):
                return_object, return_code, return_headers = f'Cannot create folder at {abs_path} - file already exists', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                os.makedirs(abs_path, exist_ok=True)
                return_object, return_code, return_headers =  f"Folder created @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}


        elif request.method=='DELETE': # trying to delete a file or folder
            if os.path.isfile(abs_path):
                try: 
                    os.remove(abs_path)
                    return_object, return_code, return_headers =     f"File deleted @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                except: return_object, return_code, return_headers = f"Cannot delete file @ {abs_path}", HTTPStatus.NOT_ACCEPTABLE, {}
            elif os.path.isdir(abs_path):
                rok = True
                if int(request.headers.get(HeaderType.XTYPE)):
                    try: rmtree(abs_path)
                    except: rok=False
                else:
                    try: os.rmdir(abs_path)
                    except: rok=False
                if rok: return_object, return_code, return_headers =   f"Folder deleted @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                else:   return_object, return_code, return_headers =   f'Cannot delete folder at {abs_path}', HTTPStatus.NOT_ACCEPTABLE, {}
            else: return_object, return_code, return_headers =         f'Cannot delete at {abs_path} - not a file or folder', HTTPStatus.NOT_ACCEPTABLE, {}

        else: return_object, return_code, return_headers =  f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST, {}

    return return_object, return_code, return_headers




#%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
start_time = datetime.datetime.now()
print('◉ start server @ [{}]'.format(start_time))
#print(f'{app.config["uids"]}')
serve(app, # https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html
    host = args.host,          
    port = args.port,          
    url_scheme = 'http',     
    threads = args.threads,    
    connection_limit = args.limit+1,
    max_request_header_size = HRsizes.tobytes(args.maxH),
    max_request_body_size = HRsizes.tobytes(args.maxB),
    
)
#<-------------------DO NOT WRITE ANY CODE AFTER THIS
end_time = datetime.datetime.now()
print('')
print('◉ stop server @ [{}]'.format(end_time))
print('◉ server up-time was [{}]'.format(end_time - start_time))



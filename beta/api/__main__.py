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
from . import HeaderType, RequestContentType, ResponseContentType, StoreType, DefaultArgs

#PYDIR = os.path.dirname(__file__) # script directory of __main__.py
try:

    from flask import Flask, request, send_file
    from waitress import serve
    from http import HTTPStatus
    from shutil import rmtree
except: exit(f'[!] Required packages missing - pip install Flask waitress')
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
DEFAULT_CONFIG_CODE = """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# known.API configuration file                                                                                   ||
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# =================================================================================================================
# the running configuration - must be a dict named as variable 'config'                                          ||
# =================================================================================================================
config = {
    'maxH'        : "0.25GB",               # (str) maximum http header size
    'maxB'        : "1.0GB",                # (str) maximum http body size
    'limit'       : 5,                      # (int) maximum connection limit to the server
    'port'        : "8888",                 # (str) server port
    'host'        : "0.0.0.0",              # (str) server address (keep 0.0.0.0 to run on all IPs)
    'allow'       : "",                     # (str) a comma-seperated list of host IP address that are allowed to POST (keep blank to allow all)
    'threads'     : 10,                     # (int) no of threads on the server
    'access'      : 'access',               # (str) the access dict variable name (as string) defined in this file 
}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Note: we can define multiple `access` dicts and then choose in running config 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# =================================================================================================================




# =================================================================================================================
# the access configuration - must be a dict specified in config['access']                                        ||
# =================================================================================================================
access = { 
#   username    :   (   access-string,      storage_path,       handle_function      ),
    'root'      :   (   'AGPD',             '!',                'handle'             ),  
}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Note: 
# by default 
#   a `root` user is created 
#   storage is set to the workdir excluding top level files
# `access` is a dict of { user_name : (access_string, storage_path, handle_function) }
# -> if access dict is blank, no users will be allowed
# -> `user_name` is a string specifying the user
#     --> `access_string` is a string specifying the access permissions of the user
#     --> `storage_path` is a string specifying absolute path which acts as a base-dir for that user 
#             ~ keep blank to use directory specified in the `--dir` argument
#             ~ prefix with '!' to exclude top-level files
#             ~ set as None or ... or anything not str to disable storage
#     --> `handle_function` is a callable like `lambda request_content, request_type: (response_content, response_tag)`
#             ~ handle_function must be define in this file as well
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# access_string can be made up of the following:
#   (A)- can access API vis POST request (at root url)
#   (G)- can download files and can list folder contents (GET)
#   (P)- can write/overwrite files and create folder (PUT, POST)
#   (D)- can delete files and folder (DELETE)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# =================================================================================================================





# =================================================================================================================
# handle functions                                                                                               ||
# =================================================================================================================
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Note: this function handles an incoming request from client - these are POST requests directly to the root url 
# use arguments `request_type` and `request_content` to access the client request data and process the data ...
# return a valid response containing a 2-tuple (response_content, response_tag)
# `response_content` should be one of (str, bytes, dict, list, tuple )
# `response_tag` is just a string
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# =================================================================================================================

def handle(request_content:object, request_type:str):
    if request_content is None : ... # request_content is None
    elif request_type == 'MESG': ... # request_content is string   (   request_content = request.get_data().decode('utf-8') )
    elif request_type == 'BYTE': ... # request_content is bytes    (   request_content = request.get_data()                 )
    elif request_type == 'FORM': ... # request_content is 2-tuple  (   request_content = (request.form, request.files)      )
    elif request_type == 'JSON': ... # request_content is json     (   request_content = request.get_json()                 )
    else                       : ... # request_content is unknown
    response_content, response_tag = "default_response", "default_tag"
    return response_content, response_tag

# =================================================================================================================

"""


#-----------------------------------------------------------------------------------------
# Parse arguments 
#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
for k,v in DefaultArgs.items(): parser.add_argument(f'--{k}', **v)
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
CONFIG = 'config'
CONFIG_MODULE = parsed.config if parsed.config else 'api' # the name of configs module
CONFIGS_FILE = f'{CONFIG_MODULE}.py' # the name of configs file


# check if 'configs.py` exsists or not`
CONFIGS_FILE_PATH = os.path.join(WORKDIR, CONFIGS_FILE) # should exsist under workdir
first_exit=False
if not os.path.isfile(CONFIGS_FILE_PATH):
    print(f'↪ Creating default config "{CONFIGS_FILE}" from ...')
    with open(CONFIGS_FILE_PATH, 'w', encoding='utf-8') as f: f.write(DEFAULT_CONFIG_CODE)
    first_exit=True
try: 
    c_module = importlib.import_module(f'{CONFIG_MODULE}')
    print(f'↪ Imported config-module "{CONFIG_MODULE}" from {c_module.__file__}')
except: exit(f'[!] Could import configs module "{CONFIG_MODULE}" at "{CONFIGS_FILE_PATH[:-3]}"')
try:
    print(f'↪ Reading config from {CONFIG_MODULE}.{CONFIG}')
    config_dict = getattr(c_module, CONFIG)
    #print(f'  ↦ type:{type(config_dict)}')
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


ACCESS = args.access
try:
    print(f'↪ Getting access dict from {CONFIG_MODULE}.{ACCESS}')
    user_access = getattr(c_module, ACCESS)
    #print(f'  ↦ type:{type(user_access)}')
except:
    exit(f'[!] Could not get access dict from {CONFIG_MODULE}.{ACCESS}')

user_handles = {}
try:
    for k,(_,_,HANDLE) in user_access.items():
        print(f'↪ Getting handle from {CONFIG_MODULE}.{HANDLE}')
        user_handles[k] = getattr(c_module, HANDLE)
        #print(f'  ↦ type:{type(user_handles[k])}')
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


def GET_ACCESS_CONFIG(access_dict):
    ac_access, ac_storage, ac_xtop = {}, {}, {}
    for user_name,(access_string,storage_path,_) in access_dict.items():
        ac_access[user_name] = access_string.upper()

        if not isinstance (storage_path, str):  ac_xtop[user_name], ac_storage[user_name] =    True,  None # no storage
        else:
            if storage_path.startswith('!'):      ac_xtop[user_name], ac_storage[user_name] =    True,  storage_path[1:]
            else:                                 ac_xtop[user_name], ac_storage[user_name] =    False, storage_path       
            if ac_storage[user_name]=='':         ac_storage[user_name] = os.path.abspath(WORKDIR) # workdir
            else:                                 ac_storage[user_name] = os.path.abspath(ac_storage[user_name]) # abspath
    return ac_access, ac_storage, ac_xtop
        



app.config['allow'] = allowed
app.config['access'], app.config['storage'], app.config['xtop']  =  GET_ACCESS_CONFIG(user_access)
#-----------------------------------------------------------------------------------------

def sprint(uid, method, path): print('↦ [{}]\t{}\t({})\t{}'.format(datetime.datetime.now(), method, uid, path))

#-----------------------------------------------------------------------------------------
# NOTE on return type
# ... type must be a string, dict, list, 
# ... type can be this but ignored: tuple with headers or status, Response instance, or WSGI callable
#-----------------------------------------------------------------------------------------
@app.route('/', methods =['GET', 'POST'])
def home():
    global user_handles
    
    if request.method == 'POST':
        request_from = request.environ['REMOTE_HOST'] 
        if request_from in app.config['allow']:

            # The clients making post request will have to provide these two headers
            # headers are only read for post requests from allowed users
            xtag, xtype = request.headers.get(HeaderType.XTAG), request.headers.get(HeaderType.XTYPE)
            if xtag not in app.config['access']: xcontent = None
            else:
                if 'A' not in app.config['access'][xtag]: xcontent = None
                elif xtype is None:             xcontent = None
                #------------------------------------------------------------------------------- Read from the reuest made by client
                elif xtype==RequestContentType.MESG: xcontent = request.get_data().decode('utf-8')
                elif xtype==RequestContentType.BYTE: xcontent = request.get_data()
                elif xtype==RequestContentType.FORM: xcontent = request.form, request.files
                elif xtype==RequestContentType.JSON: xcontent = request.get_json()
                #-------------------------------------------------------------------------------
                else:                         xcontent = None               
            
            if xcontent is not None:
                return_object, return_tag = user_handles[xtag](xcontent, xtype)
                return_type = ResponseContentType.ALL.get(type(return_object), None)
                if return_type is not None: 
                    return_code = HTTPStatus.OK
                    return_headers = {HeaderType.XTAG :f'{return_tag}', HeaderType.XTYPE:return_type} #<-- headers are only sent when content and types are valid
                    sprint(xtag, 'API', xtype)
                else:   return_object, return_code, return_headers = f"[!] Invalid response from handler [{type(return_object)}::{return_type}:{return_tag}]", HTTPStatus.NOT_FOUND, {}
            else:       return_object, return_code, return_headers = f'[!] Type "{xtype}" is not a valid content type', HTTPStatus.NOT_ACCEPTABLE, {}
        else:           return_object, return_code, return_headers = f"[!] You are not allowed to POST", HTTPStatus.NOT_ACCEPTABLE, {}
    elif request.method == 'GET':     
        return_object = f'<pre>[Known.api]@{__file__}\n[Workdir]@{WORKDIR}\n[Config]@{CONFIGS_FILE_PATH}</pre>'
        #for k,v in args._get_kwargs(): return_object+=f'\n\t{k}\t{v}\n'
        return_code, return_headers = HTTPStatus.OK, {}
    else: return_object, return_code, return_headers = f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST, {}
    
    return return_object, return_code, return_headers


# Storage urls for file-storage api
# tag specifies a name of file, type specifies if its a overall-view, a directory listing or a file

@app.route('/store', methods =['GET'])
def storageview(): # an overview of all storage paths and the files in them
    uid = request.headers.get(HeaderType.XTAG)
    if uid not in app.config['access']:  return f'Invalid client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
    else:
        if 'G' not in app.config['access'][uid]:  return_object, return_code, return_headers = {}, HTTPStatus.NOT_ACCEPTABLE, {}
        else:
            basedir = app.config['storage'][uid]
            if basedir is None: return_object, return_code, return_headers = {}, HTTPStatus.NOT_FOUND, {}
            else: return_object, return_code, return_headers =  {os.path.relpath(root, basedir) : files for root, directories, files in os.walk(basedir)}, HTTPStatus.OK, {HeaderType.XTAG: f'{basedir}', HeaderType.XTYPE: StoreType.HOME}
        return return_object, return_code, return_headers

@app.route('/store/', methods =['GET'])
def storageroot(): # root dir
    uid = request.headers.get(HeaderType.XTAG)
    if uid not in app.config['access']:  return f'Invalid client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
    else:
        if 'G' not in app.config['access'][uid]:  return_object, return_code, return_headers = {}, HTTPStatus.NOT_ACCEPTABLE, {}
        else:
            basedir = app.config['storage'][uid]
            print(f'{basedir=}')
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

    uid = request.headers.get(HeaderType.XTAG)
    if uid not in app.config['access']:  return f'Invalid client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
    basedir = app.config['storage'][uid]
    if basedir is None: return_object, return_code, return_headers = f'No Storage is specified for Client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
    else:
        access_string = app.config['access'][uid]
        abs_path = os.path.join(basedir, req_path) # Joining the base and the requested path
        rel_path = os.path.relpath(abs_path, basedir)
        if request.method=='GET': # trying to download that file or view a directory
            if 'G' not in access_string: return_object, return_code, return_headers = f'GET Access denined to client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                if os.path.exists(abs_path):
                    if os.path.isdir(abs_path):     
                        rw, dw, fw = next(iter(os.walk(abs_path)))
                        fs = [os.path.getsize(os.path.join(rw, fwi)) for fwi in fw]
                        return_object = dict(base=rel_path, folders=dw, files={k:round(v/1024,2) for k,v in zip(fw,fs)}) # size in KB
                        return_code = HTTPStatus.OK
                        return_headers = {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.DIR}
                    else: 
                        # make sure to exclude top level files
                        if app.config['xtop'][uid]:
                            if (basedir==os.path.dirname(abs_path)): 
                                return f'Top-Level Path is excluded: {abs_path} - cannot read', HTTPStatus.NOT_ACCEPTABLE, {} #<-----RETURNING HERE

                        resx = send_file(abs_path) 
                        resx.headers[HeaderType.XTAG] = os.path.basename(abs_path) # 'save_as'
                        resx.headers[HeaderType.XTYPE] = StoreType.FILE
                        sprint(uid, 'STORE:DN', abs_path) # downloading a file 
                        return resx #<-----RETURNING HERE
                    
                else: return_object, return_code, return_headers = f'Path not found: {abs_path}', HTTPStatus.NOT_FOUND, {}


        elif request.method=='POST': # trying to create new file or replace existing file
            if 'P' not in access_string: return_object, return_code, return_headers = f'POST Access denined to client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                if os.path.isdir(abs_path):
                    return_object, return_code, return_headers = f'Cannot create file # {abs_path} - folder already exists', HTTPStatus.NOT_ACCEPTABLE, {}
                else:
                    if app.config['xtop'][uid]:
                        if (basedir==os.path.dirname(abs_path)): 
                            return f'Top-Level Path is excluded: {abs_path} - cannot write', HTTPStatus.NOT_ACCEPTABLE, {} #<-----RETURNING HERE
                    try: # overwrite
                        with open(abs_path, 'wb') as f: f.write(request.get_data())
                        return_object, return_code, return_headers =  f"File created @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                        sprint(uid, 'STORE:UP', abs_path) # uploading a file
                    except: return_object, return_code, return_headers =   f"Cannot create file @ {abs_path}", HTTPStatus.NOT_ACCEPTABLE, {}


        elif request.method=='PUT': # trying to create new directory
            if 'P' not in access_string: return_object, return_code, return_headers = f'PUT Access denined to client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                if os.path.isfile(abs_path):
                    return_object, return_code, return_headers = f'Cannot create folder at {abs_path} - file already exists', HTTPStatus.NOT_ACCEPTABLE, {}
                else:
                    os.makedirs(abs_path, exist_ok=True)
                    return_object, return_code, return_headers =  f"Folder created @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                    sprint(uid, 'STORE:MK', abs_path) # creating a folder


        elif request.method=='DELETE': # trying to delete a file or folder
            if 'D' not in access_string: return_object, return_code, return_headers = f'DELETE Access denined to client uid [{uid}]', HTTPStatus.NOT_ACCEPTABLE, {}
            else:
                if os.path.isfile(abs_path):
                    if app.config['xtop'][uid]:
                        if (basedir==os.path.dirname(abs_path)): 
                            return f'Top-Level Path is excluded: {abs_path} - cannot delete', HTTPStatus.NOT_ACCEPTABLE, {} #<-----RETURNING HERE
                    try: 
                        os.remove(abs_path)
                        return_object, return_code, return_headers =     f"File deleted @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                        sprint(uid, 'STORE:DF', abs_path) # removing a file
                    except: return_object, return_code, return_headers = f"Cannot delete file @ {abs_path}", HTTPStatus.NOT_ACCEPTABLE, {}
                elif os.path.isdir(abs_path):
                    rok = True
                    rhead = request.headers.get(HeaderType.XTYPE)
                    if rhead is None: rhead = 0 # default behaviour is to not go for recursive delete
                    if int(rhead):
                        try: 
                            rmtree(abs_path)
                            sprint(uid, 'STORE:DR', abs_path) # removing a directory (recursive)
                        except: rok=False
                    else:
                        try: 
                            os.rmdir(abs_path)
                            sprint(uid, 'STORE:DE', abs_path) # removing a directory (empty)
                        except: rok=False
                    if rok: 
                        return_object, return_code, return_headers =   f"Folder deleted @ {abs_path}", HTTPStatus.OK, {HeaderType.XTAG: f'{rel_path}', HeaderType.XTYPE: StoreType.MSG}
                        
                    else:   return_object, return_code, return_headers =   f'Cannot delete folder at {abs_path}', HTTPStatus.NOT_ACCEPTABLE, {}
                else: return_object, return_code, return_headers =         f'Cannot delete at {abs_path} - not a file or folder', HTTPStatus.NOT_ACCEPTABLE, {}

        else: return_object, return_code, return_headers =  f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST, {}

    return return_object, return_code, return_headers




#%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
start_time = datetime.datetime.now()
print('◉ start server @ [{}]'.format(start_time))
# print(f'{app.config["access"]}')
# print(f'{app.config["storage"]}')
# print(f'{app.config["xtop"]}')
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



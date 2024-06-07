__doc__="""
HTTP API Request-Response model

Client applications will use only the 'POST' method which can be called with 3 types of data

response = requests.post(
    url=server,
    data={key: value}, # A dictionary, list of tuples, bytes or a file object to send to the specified url
    json={key: value}, # A JSON object to send to the specified url
    files={key: value}, # File handles (opened)
    timeout=0.0,
    )

The 'GET' method will be used as 

response = requests.get(
    url=server,
    params={key: value}, # A dictionary of string params in a get url like scheme
    timeout=0.0,
    )


How to design User Module to handle api events

# ----------------------------------------------------------------------
# as global function 
# python -m known.api --user=user.py 
# ----------------------------------------------------------------------
def handle_json(host, data):
    f = f'↪ [Module handle json]\n{host=}\n{data=}\n'
    print(f)
    return f


def handle_file(host, files):
    f = f'↪ [Module handle file]\n{host=}\n{files=}\n'
    print(f)
    return f


# ----------------------------------------------------------------------
# as a global object 
# python -m known.api --user=user.py --object=StaticChan
# ----------------------------------------------------------------------
class StaticChan:

    @staticmethod
    def handle_json(host, data):
        f = f'↪ [{__class__} Module handle json]\n{host=}\n{data=}\n'
        print(f)
        return f
    
    @staticmethod
    def handle_file(host, files):
        f = f'↪ [{__class__} Module handle file]\n{host=}\n{files=}\n'
        print(f)
        return f


# ----------------------------------------------------------------------
# as a global Module (callable - requires initialization) 
# python -m known.api --user=user.py --object=InitChan --callable=1
# ----------------------------------------------------------------------
class InitChan:

    def __init__(self): pass

    def handle_json(self, host, data):
        f = f'↪ [{self.__class__} Module handle json]\n{host=}\n{data=}\n'
        print(f)
        return f
    
    
    def handle_file(self, host, files):
        f = f'↪ [{self.__class__} Module handle file]\n{host=}\n{files=}\n'
        print(f)
        return f
"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# imports ----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
import os, argparse, datetime, importlib, importlib.util

#PYDIR = os.path.dirname(__file__) # script directory of __main__.py
try:
    from flask import Flask, request
    from waitress import serve
    from http import HTTPStatus
except: exit(f'[!] The required packages missing:\tFlask>=3.0.2, waitress>=3.0.0\n  ⇒ pip install Flask waitress')


class DefaultUserModule:
    def handle_args(self, host, args):        return f"handle_args({host}, {args})"
    def handle_json(self, host, data):        return f"handle_json({host}, {data})"    
    def handle_file(self, host, files):       return f"handle_file({host}, {files})"


parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default='', help="path of main user module file")
parser.add_argument('--object', type=str, default='', help="path of main user module")
parser.add_argument('--callable', type=int, default=0, help="if true, calls the class to create a new instance (with no args)")
parser.add_argument('--handle_json', type=str, default='handle_json', help="path of main user module file")
parser.add_argument('--handle_file', type=str, default='handle_file', help="path of main user module file")
parser.add_argument('--handle_args', type=str, default='handle_args', help="path of main user module file")
parser.add_argument('--host', type=str, default='0.0.0.0', help="host ip-address")
parser.add_argument('--port', type=str, default='8080', help="host port")
parser.add_argument('--max', type=str, default='1GB', help="max_request_body_size")
parser.add_argument('--limit', type=int, default=10, help="connection limit")
parsed = parser.parse_args()

if parsed.user:
    # user-module
    USER_MODULE_FILE_PATH = os.path.abspath(parsed.user)
    print(f'↪ Loading user-module from {USER_MODULE_FILE_PATH}')
    if not os.path.isfile(USER_MODULE_FILE_PATH): exit(f'Invalid user-module file @ {USER_MODULE_FILE_PATH}')
    try: 
        # from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
        user_module_spec = importlib.util.spec_from_file_location("", USER_MODULE_FILE_PATH)
        user_module = importlib.util.module_from_spec(user_module_spec)
        user_module_spec.loader.exec_module(user_module)
        print(f'↪ Imported user-module from {user_module.__file__}')
    except: exit(f'[!] Could import user-module "{USER_MODULE_FILE_PATH}"')
    if parsed.object:
        try:
            user_module = getattr(user_module, parsed.object)
            if bool(parsed.callable): user_module = user_module()
        except:
            exit(f'Could not load object {parsed.object}')
    USER_HANDLE_JSON = parsed.handle_json
    USER_HANDLE_FILE = parsed.handle_file
    USER_HANDLE_ARGS = parsed.handle_args
else:
    print(f'↪ [!] user-module not defined, using default.')
    USER_HANDLE_JSON = 'handle_json'
    USER_HANDLE_FILE = 'handle_file'
    USER_HANDLE_ARGS = 'handle_args'
    user_module = DefaultUserModule()

if not hasattr(user_module, USER_HANDLE_ARGS): exit(f'[!] ARGS Handler Method not found {user_module}.{USER_HANDLE_ARGS}')
if not hasattr(user_module, USER_HANDLE_JSON): exit(f'[!] JSON Handler Method not found {user_module}.{USER_HANDLE_JSON}')
if not hasattr(user_module, USER_HANDLE_FILE): exit(f'[!] FILE Handler Method not found {user_module}.{USER_HANDLE_FILE}')
#from known.basic import Verbose as vb
# ------------------------------------------------------------------------------------------
# application setting and instance
# ------------------------------------------------------------------------------------------
app = Flask(__name__)
#app.secret_key =         'api_test'
#app.config['key'] =      'value'

# NOTE: "return_object" must be a string, dict, list, tuple with headers or status, Response instance, or WSGI callable

def is_valid_return_object(x): return isinstance(x, (str, dict, list))

@app.route('/', methods =['GET', 'POST'])
def home():
    #vb.show(request)
    if request.method == 'POST':
        #print('FILENAME:', request.files.filename())
        if request.is_json:  
            return_object = getattr(user_module, USER_HANDLE_JSON)(request.host, request.get_json())           
            if is_valid_return_object(return_object): return_code = HTTPStatus.OK
            else: return_object, return_code = f"[!] [Handle-POST-JSON] The Return Type {type(return_object)} is Invalid", HTTPStatus.NOT_FOUND
        else:
            if request.args or request.form: 
                return_object = getattr(user_module, USER_HANDLE_ARGS)(request.host, request.form)   
                if is_valid_return_object(return_object): return_code = HTTPStatus.OK
                else: return_object, return_code = f"[!] [Handle-POST-ARGS] The Return Type {type(return_object)} is Invalid", HTTPStatus.NOT_FOUND
            else:
                if request.files: 
                    return_object = getattr(user_module, USER_HANDLE_FILE)(request.host, request.files) 
                    if is_valid_return_object(return_object): return_code = HTTPStatus.OK
                    else: return_object, return_code = f"[!] [Handle-POST-FILE] The Return Type {type(return_object)} is Invalid", HTTPStatus.NOT_FOUND
                else: return_object, return_code = f"[!] Invalid [POST] Request - Must be json or have files", HTTPStatus.BAD_REQUEST
    elif request.method == 'GET':       
        return_object = f'<pre>[Known.api]@{__file__}\n'
        for k,v in parsed._get_kwargs(): return_object+=f'\n\t{k}\t{v}\n'
        return_object+='</pre>'
        return_code = HTTPStatus.OK
    else:                               return_object, return_code = f"[!] Invalid Request Type {request.method}", HTTPStatus.BAD_REQUEST
    # only use either one of data or json in post request (not both)
    return return_object, return_code

   

#%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#<-------------------DO NOT WRITE ANY CODE AFTER THIS

HRsizes = dict(KB=2**10, MB=2**20, GB=2**30, TB=2**40)
str2bytes = lambda size: int(float(size[:-2])*HRsizes.get(size[-2:].upper(), 0))

start_time = datetime.datetime.now()
print('◉ start server @ [{}]'.format(start_time))
serve(app, # https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html
    host = parsed.host,          
    port = parsed.port,          
    url_scheme = 'http',     
    threads = 1,    
    connection_limit = parsed.limit,
    max_request_body_size = str2bytes(parsed.max),
)
#<-------------------DO NOT WRITE ANY CODE AFTER THIS
end_time = datetime.datetime.now()
print('◉ stop server @ [{}]'.format(end_time))
print('◉ server up-time was [{}]'.format(end_time - start_time))

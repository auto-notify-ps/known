__doc__=f""" 
-------------------------------------------------------------
Fly - Flask-based web app for sharing files 
-------------------------------------------------------------
"""
#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------

#%% [PRE-INITIALIZATION] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 

import argparse
# ------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# python -m known.fly --help
parser.add_argument('--dir', type=str, default='', help="path of workspace directory")
parser.add_argument('--verbose', type=int, default=2, help="verbose level in logging")
parser.add_argument('--log', type=str, default='', help="name of logfile as date-time-formated string e.g. fly_%Y_%m_%d_%H_%M_%S_%f_log.txt [Note: keep blank to disable logging]")
parser.add_argument('--con', type=str, default='', help="config name - if not provided, uses 'default'")
parser.add_argument('--reg', type=str, default='', help="if specified, allow users to register with specified access string such as DABU or DABUS+")
parser.add_argument('--cos', type=int, default=1, help="use 1 to create-on-start - create (overwrites) pages")
parser.add_argument('--coe', type=int, default=0, help="use 1 to clean-on-exit - deletes pages")
parser.add_argument('--access', type=str, default='', help="if specified, adds extra premissions to access string (catenates) for this session only")
parser.add_argument('--msl', type=int, default=100, help="Max String Length for UID/NAME/PASSWORDS")
parser.add_argument('--eip', type=int, default=1, help="Evaluate Immediate Persis. If True, persist the eval-db after each single evaluation (eval-db in always persisted after update from template)")
parsed = parser.parse_args()
# ------------------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------------------
import os, re, getpass, random, logging, importlib.util
from io import BytesIO
from math import inf
import datetime
def fnow(format): return datetime.datetime.strftime(datetime.datetime.now(), format)
try:
    from flask import Flask, render_template, request, redirect, url_for, session, abort, send_file
    from flask_wtf import FlaskForm
    from wtforms import SubmitField, MultipleFileField
    from werkzeug.utils import secure_filename
    from wtforms.validators import InputRequired
    from waitress import serve
except: exit(f'[!] The required Flask packages missing:\tFlask>=3.0.2, Flask-WTF>=1.2.1\twaitress>=3.0.0\n  ⇒ pip install Flask Flask-WTF waitress')
try: 
    from nbconvert import HTMLExporter 
    has_nbconvert_package=True
except:
    print(f'[!] IPYNB to HTML rending will not work since nbconvert>=7.16.2 is missing\n  ⇒ pip install nbconvert')
    has_nbconvert_package = False
# ------------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------------
LOGF = f'{parsed.log}' 
LOGFILE = None
if LOGF and parsed.verbose>0: 
    LOGFILENAME = f'{fnow(LOGF)}'
    try:# Set up logging to a file # also output to the console
        LOGFILE = os.path.abspath(LOGFILENAME)
        logging.basicConfig(filename=LOGFILE, level=logging.INFO, format='%(asctime)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    except: exit(f'[!] Logging could not be setup at {LOGFILE}')
# ------------------------------------------------------------------------------------------
# verbose level
# ------------------------------------------------------------------------------------------
if parsed.verbose==0: # no log
    def sprint(msg): pass
    def dprint(msg): pass
    def fexit(msg): exit(msg)
elif parsed.verbose==1: # only server logs
    if LOGFILE is None:
        def sprint(msg): print(msg) 
        def dprint(msg): pass 
        def fexit(msg): exit(msg)
    else:
        def sprint(msg): logging.info(msg) 
        def dprint(msg): pass 
        def fexit(msg):
            logging.error(msg) 
            exit()
elif parsed.verbose>=2: # server and user logs
    if LOGFILE is None:
        def sprint(msg): print(msg) 
        def dprint(msg): print(msg) 
        def fexit(msg): exit(msg)
    else:
        def sprint(msg): logging.info(msg) 
        def dprint(msg): logging.info(msg) 
        def fexit(msg):
            logging.error(msg) 
            exit()
else: raise ZeroDivisionError # impossible
# ------------------------------------------------------------------------------------------


#%% [INITIALIZATION] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# ------------------------------------------------------------------------------------------
sprint(f'Starting...')
sprint(f'↪ Logging @ {LOGFILE}')
# ------------------------------------------------------------------------------------------
# workdir
#-----------------------------------------------------------------------------------------
WORKDIR = f'{parsed.dir}' # define working dir - contains all bases
if not WORKDIR: WORKDIR = os.getcwd()
WORKDIR=os.path.abspath(WORKDIR)
try: os.makedirs(WORKDIR, exist_ok=True)
except: fexit(f'[!] Workspace directory was not found and could not be created')
sprint(f'↪ Workspace directory is {WORKDIR}')
#-----------------------------------------------------------------------------------------
# globals
#-----------------------------------------------------------------------------------------
CSV_DELIM = ','
SSV_DELIM = '\n'
NEWLINE = '\n'
TABLINE = '\t'
LOGIN_ORD = ['ADMIN','UID','NAME','PASS']
LOGIN_ORD_MAPPING = {v:i for i,v in enumerate(LOGIN_ORD)}
EVAL_ORD = ['UID', 'NAME', 'SCORE', 'REMARK', 'BY']
DEFAULT_USER = 'admin'
DEFAULT_ACCESS = f'DABUSRX+-'

MAX_STR_LEN = int(parsed.msl) if parsed.msl>0 else 1
def rematch(instr, pattern):  return \
    (len(instr) >= 0) and \
    (len(instr) <= MAX_STR_LEN) and \
    (re.match(pattern, instr))

def VALIDATE_PASS(instr):     return rematch(instr, r'^[a-zA-Z0-9~!@#$%^&*()_+{}<>?`\-=\[\].]+$')
def VALIDATE_UID(instr):      return rematch(instr, r'^[a-zA-Z0-9._@]+$') and instr[0]!="."
def VALIDATE_NAME(instr):     return rematch(instr, r'^[a-zA-Z0-9]+(?: [a-zA-Z0-9]+)*$')

def DICT2CSV(path, d, ord):
    with open(path, 'w', encoding='utf-8') as f: 
        f.write(CSV_DELIM.join(ord)+SSV_DELIM)
        for v in d.values(): f.write(CSV_DELIM.join(v)+SSV_DELIM)

def DICT2BUFF(d, ord):
    b = BytesIO()
    b.write(f'{CSV_DELIM.join(ord)+SSV_DELIM}'.encode(encoding='utf-8'))
    for v in d.values(): b.write(f'{CSV_DELIM.join(v)+SSV_DELIM}'.encode(encoding='utf-8'))
    b.seek(0)
    return b

def S2DICT(s, key_at):
    lines = s.split(SSV_DELIM)
    d = dict()
    for line in lines[1:]:
        if line:
            cells = line.split(CSV_DELIM)
            d[f'{cells[key_at]}'] = cells
    return d

def CSV2DICT(path, key_at):
    with open(path, 'r', encoding='utf-8') as f: s = f.read()
    return S2DICT(s, key_at)

def BUFF2DICT(b, key_at):
    b.seek(0)
    return S2DICT(b.read().decode(encoding='utf-8'), key_at)

def GET_SECRET_KEY(postfix):
    randx = lambda : random.randint(1111111111, 9999999999)
    r1 = randx()
    for _ in range(datetime.datetime.now().microsecond % 60): _ = randx()
    r2 = randx()
    for _ in range(datetime.datetime.now().second): _ = randx()
    r3 = randx()
    for _ in range(datetime.datetime.now().minute): _ = randx()
    r4 = randx()
    return ':{}:{}:{}:{}:{}:'.format(r1,r2,r3,r4,postfix)

def READ_DB_FROM_DISK(path, key_at):
    try:    return CSV2DICT(path, key_at), True
    except: return dict(), False

def WRITE_DB_TO_DISK(path, db_frame, ord): # will change the order
    try:
        DICT2CSV(path, db_frame, ord) # save updated login information to csv
        return True
    except PermissionError:
        return False

def GET_FILE_LIST (d): 
    dlist = []
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p): dlist.append(f)
    return sorted(dlist)

def DISPLAY_SIZE_READABLE(mus):
    # find max upload size in appropiate units
    mus_kb = mus/(2**10)
    if len(f'{int(mus_kb)}') < 4:
        mus_display = f'{mus_kb:.2f} KB'
    else:
        mus_mb = mus/(2**20)
        if len(f'{int(mus_mb)}') < 4:
            mus_display = f'{mus_mb:.2f} MB'
        else:
            mus_gb = mus/(2**30)
            if len(f'{int(mus_gb)}') < 4:
                mus_display = f'{mus_gb:.2f} GB'
            else:
                mus_tb = mus/(2**40)
                mus_display = f'{mus_tb:.2f} TB'
    return mus_display

def NEW_NOTEBOOK_STR(title, nbformat=4, nbformat_minor=2):
    return '{"cells": [{"cell_type": "markdown","metadata": {},"source": [ "'+str(title)+'" ] } ], "metadata": { }, "nbformat": '+str(nbformat)+', "nbformat_minor": '+str(nbformat_minor)+'}'

class Fake:
    def __len__(self): return len(self.__dict__)
    def __init__(self, **kwargs) -> None:
        for name, attribute in kwargs.items():  setattr(self, name, attribute)
#-----------------------------------------------------------------------------------------
















#-----------------------------------------------------------------------------------------
# ==> default configurations
#-----------------------------------------------------------------------------------------

def DEFAULT_CONFIG(file_path):
    with open(file_path, 'w', encoding='utf-8') as f: f.write("""

def merged(a:dict, b:dict): return {**a, **b}

default = dict(    

    # -------------------------------------# general info
    topic        = "Fly",             # topic text (main banner text)
    welcome      = "Login to Continue",    # msg shown on login page
    register     = "Register User",        # msg shown on register (new-user) page
    emoji        = "🦋",                   # emoji shown of login page and seperates uid - name
    rename       = 0,                      # if rename=1, allows users to update their names when logging in
    repass       = 1,                      # if repass=1, allows admins and Xs to reset passwords for users - should be enabled in only one session (for multi-session)
    case         = 0,                      # case-sentivity level in uid
                                            #   (if case=0 uids are not converted           when matching in database)
                                            #   (if case>0 uids are converted to upper-case when matching in database)
                                            #   (if case<0 uids are converted to lower-case when matching in database)
    
    # -------------------------------------# validation
    required     = "",                     # csv list of file-names that are required to be uploaded e.g., required = "a.pdf,b.png,c.exe" (keep blank to allow all file-names)
    extra        = 1,                      # if true, allows uploading extra file (other tna required)
    maxupcount   = -1,                     # maximum number of files that can be uploaded by a user (keep -1 for no limit and 0 to disable uploading)
    maxupsize    = "40GB",                 # maximum size of uploaded file (html_body_size)
    
    # -------------------------------------# server config
    maxconnect   = 50,                     # maximum number of connections allowed to the server
    threads      = 4,                      # no. of threads used by waitress server
    port         = "8888",                 # port
    host         = "0.0.0.0",              # ip

    # ------------------------------------# file and directory information
    base         = "__base__",            # the base directory 
    html         = "__pycache__",         # use pycache dir to store flask html
    secret       = "secret.txt",      # flask app secret
    login        = "login.csv",       # login database
    eval         = "eval.csv",        # evaluation database - created if not existing - reloads if exists
    uploads      = "uploads",         # uploads folder (uploaded files by users go here)
    reports      = "reports",         # reports folder (personal user access files by users go here)
    downloads    = "downloads",       # downloads folder
    store        = "store",           # store folder
    board        = "board.ipynb",     # board file
    # --------------------------------------# style dict
    style        = dict(                   
                        # -------------# labels
                        downloads_ =    'Downloads',
                        uploads_ =      'Uploads',
                        store_ =        'Store',
                        board_=         'Board',
                        logout_=        'Logout',
                        login_=         'Login',
                        new_=           'Register',
                        eval_=          'Eval',
                        resetpass_=     'Reset',
                        report_=        'Report',

                        # -------------# colors 
                        bgcolor      = "white",
                        fgcolor      = "black",
                        refcolor     = "#232323",
                        item_bgcolor = "#232323",
                        item_normal  = "#e6e6e6",
                        item_true    = "#47ff6f",
                        item_false   = "#ff6565",
                        flup_bgcolor = "#ebebeb",
                        flup_fgcolor = "#232323",
                        fldown_bgcolor = "#ebebeb",
                        fldown_fgcolor = "#232323",
                        msgcolor =     "#060472",
                        
                        # -------------# icons 
                        icon_board =    '🔰',
                        icon_login=     '🔒',
                        icon_new=       '👤',
                        icon_home=      '🔘',
                        icon_downloads= '📥',
                        icon_uploads=   '📤',
                        icon_store=     '📦',
                        icon_eval=      '✴️',
                        icon_report=    '📜',
                        icon_getfile=   '⬇️',
                        icon_gethtml=   '🌐',
                        icon_hidden=    '👁️',

                        # -------------# board style ('lab'  'classic' 'reveal')
                        template_board = 'lab', 
                    )
    )

""")
        

#-----------------------------------------------------------------------------------------
# ==> read configurations
#-----------------------------------------------------------------------------------------
CONFIG = parsed.con if parsed.con else 'default' # the config-dict to read from
CONFIG_MODULE = '__configs__'  # the name of configs module
CONFIGS_FILE = f'{CONFIG_MODULE}.py' # the name of configs file
# try to import configs
CONFIGS_FILE_PATH = os.path.join(WORKDIR, CONFIGS_FILE) # should exsist under workdir
if not os.path.isfile(CONFIGS_FILE_PATH):
    sprint(f'↪ Creating default config "{CONFIGS_FILE}" ...')
    try: 
        DEFAULT_CONFIG(CONFIGS_FILE_PATH)
        sprint(f'⇒ Created new config "{CONFIG_MODULE}" at "{CONFIGS_FILE_PATH}"')
        raise AssertionError
    except AssertionError: fexit(f'⇒ Server will not start on this run, edit the config and start again')
    except: fexit(f'[!] Could find or create config "{CONFIG_MODULE}" at "{CONFIGS_FILE_PATH}"')
try: 
    # Load the module from the specified file path
    c_spec = importlib.util.spec_from_file_location(CONFIG_MODULE, CONFIGS_FILE_PATH)
    c_module = importlib.util.module_from_spec(c_spec)
    c_spec.loader.exec_module(c_module)
    sprint(f'↪ Imported config-module "{CONFIG_MODULE}" from {c_module.__file__}')
except: fexit(f'[!] Could not import configs module "{CONFIG_MODULE}" at "{CONFIGS_FILE_PATH[:-3]}"')
try:
    sprint(f'↪ Reading config from {CONFIG_MODULE}.{CONFIG}')
    if "." in CONFIG: 
        CONFIGX = CONFIG.split(".")
        config_dict = c_module
        while CONFIGX:
            m = CONFIGX.pop(0).strip()
            if not m: continue
            config_dict = getattr(config_dict, m)
    else: config_dict = getattr(c_module, CONFIG)
except:
    fexit(f'[!] Could not read config from {CONFIG_MODULE}.{CONFIG}')

if not isinstance(config_dict, dict): 
    try: config_dict=config_dict()
    except: pass
if not isinstance(config_dict, dict): raise fexit(f'Expecting a dict object for config')

try: 
    sprint(f'↪ Building config from {CONFIG_MODULE}.{CONFIG}')
    args = Fake(**config_dict)
except: fexit(f'[!] Could not read config')
if not len(args): fexit(f'[!] Empty or Invalid config provided')









































#-----------------------------------------------------------------------------------------
# Directories
#-----------------------------------------------------------------------------------------
HTMLDIR = ((os.path.join(WORKDIR, args.html)) if args.html else WORKDIR)
try: os.makedirs(HTMLDIR, exist_ok=True)
except: fexit(f'[!] HTML directory was not found and could not be created')
sprint(f'⚙ HTML Directory @ {HTMLDIR}')

BASEDIR = ((os.path.join(WORKDIR, args.base)) if args.base else WORKDIR)
try:     os.makedirs(BASEDIR, exist_ok=True)
except:  fexit(f'[!] base directory  @ {BASEDIR} was not found and could not be created') 
sprint(f'⚙ Base Directory: {BASEDIR}')

# ------------------------------------------------------------------------------------------
# WEB-SERVER INFORMATION
# ------------------------------------------------------------------------------------------\
if not args.secret: 
    APP_SECRET_KEY =  GET_SECRET_KEY(fnow("%Y%m%d%H%M%S"))
    sprint(f'⇒ secret not provided - using random secret')
else:
    APP_SECRET_KEY_FILE = os.path.join(BASEDIR, args.secret)
    if not os.path.isfile(APP_SECRET_KEY_FILE): #< --- if key dont exist, create it
        APP_SECRET_KEY =  GET_SECRET_KEY(fnow("%Y%m%d%H%M%S"))
        try:
            with open(APP_SECRET_KEY_FILE, 'w') as f: f.write(APP_SECRET_KEY) #<---- auto-generated key
        except: fexit(f'[!] could not create secret key @ {APP_SECRET_KEY_FILE}')
        sprint(f'⇒ New secret created: {APP_SECRET_KEY_FILE}')
    else:
        try:
            with open(APP_SECRET_KEY_FILE, 'r') as f: APP_SECRET_KEY = f.read()
            sprint(f'⇒ Loaded secret file: {APP_SECRET_KEY_FILE}')
        except: fexit(f'[!] could not read secret key @ {APP_SECRET_KEY_FILE}')

# ------------------------------------------------------------------------------------------
# LOGIN DATABASE - CSV
# ------------------------------------------------------------------------------------------
if not args.login: fexit(f'[!] login file was not provided!')    
LOGIN_XL_PATH = os.path.join( BASEDIR, args.login) 
if not os.path.isfile(LOGIN_XL_PATH): 
    sprint(f'⇒ Creating new login file: {LOGIN_XL_PATH}')
    
    this_user = getpass.getuser()
    if not (VALIDATE_UID(this_user)):  this_user=DEFAULT_USER

    
    try:this_name = os.uname().nodename
    except:this_name = ""
    if not (VALIDATE_NAME(this_name)):  this_name=this_user.upper()

    DICT2CSV(LOGIN_XL_PATH, 
             { f'{this_user}' : [DEFAULT_ACCESS,  f'{this_user}', f'{this_name}', f''] }, 
             LOGIN_ORD ) # save updated login information to csv
    
    sprint(f'⇒ Created new login-db with admin-user: user-id "{this_user}" and name "{this_name}"')

# ------------------------------------------------------------------------------------------
# EVAL DATABASE - CSV
# ------------------------------------------------------------------------------------------
if not args.eval: EVAL_XL_PATH = None # fexit(f'[!] evaluation file was not provided!')    
else: EVAL_XL_PATH = os.path.join( BASEDIR, args.eval)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# download settings
# ------------------------------------------------------------------------------------------
if not args.downloads: fexit(f'[!] downloads folder was not provided!')
DOWNLOAD_FOLDER_PATH = os.path.join( BASEDIR, args.downloads) 
try: os.makedirs(DOWNLOAD_FOLDER_PATH, exist_ok=True)
except: fexit(f'[!] downloads folder @ {DOWNLOAD_FOLDER_PATH} was not found and could not be created')
sprint(f'⚙ Download Folder: {DOWNLOAD_FOLDER_PATH}') 
# ------------------------------------------------------------------------------------------
# store settings
# ------------------------------------------------------------------------------------------
if not args.store: fexit(f'[!] store folder was not provided!')
STORE_FOLDER_PATH = os.path.join( BASEDIR, args.store) 
try: os.makedirs(STORE_FOLDER_PATH, exist_ok=True)
except: fexit(f'[!] store folder @ {STORE_FOLDER_PATH} was not found and could not be created')
sprint(f'⚙ Store Folder: {STORE_FOLDER_PATH}')
# ------------------------------------------------------------------------------------------
# upload settings
# ------------------------------------------------------------------------------------------
if not args.uploads: fexit(f'[!] uploads folder was not provided!')
UPLOAD_FOLDER_PATH = os.path.join( BASEDIR, args.uploads ) 
try: os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)
except: fexit(f'[!] uploads folder @ {UPLOAD_FOLDER_PATH} was not found and could not be created')
sprint(f'⚙ Upload Folder: {UPLOAD_FOLDER_PATH}')
# ------------------------------------------------------------------------------------------
# report settings
# ------------------------------------------------------------------------------------------
if not args.reports: fexit(f'[!] reports folder was not provided!')
REPORT_FOLDER_PATH = os.path.join( BASEDIR, args.reports ) 
try: os.makedirs(REPORT_FOLDER_PATH, exist_ok=True)
except: fexit(f'[!] reports folder @ {REPORT_FOLDER_PATH} was not found and could not be created')
sprint(f'⚙ Reports Folder: {REPORT_FOLDER_PATH}')

#-----------------------------------------------------------------------------------------
# file-name and uploads validation
#-----------------------------------------------------------------------------------------
ALLOWED_EXTRA = bool(args.extra)
REQUIRED_FILES = set([x.strip() for x in args.required.split(',') if x])  # a set or list of file extensions that are required to be uploaded 
if '' in REQUIRED_FILES: REQUIRED_FILES.remove('')
# def VALIDATE_FILENAME(instr):           return rematch(instr, r'^[a-zA-Z]+(?: [a-zA-Z]+)*$')
# def VALIDATE_FILENAME_SUBMIT(instr):     return rematch(instr, r'^[a-zA-Z]+(?: [a-zA-Z]+)*$')
def VALIDATE_FILENAME(filename):   # a function that checks for valid file 
    if '.' in filename: 
        name, ext = filename.rsplit('.', 1)
        safename = f'{name}.{ext.lower()}'
        if REQUIRED_FILES:  isvalid = bool(safename) if ALLOWED_EXTRA else (safename in REQUIRED_FILES)
        else:               isvalid = bool(safename) #re.match(VALID_FILES_PATTERN, safename, re.IGNORECASE)  # Case-insensitive matching
    else:               
        name, ext = filename, ''
        safename = f'{name}'
        if REQUIRED_FILES:  isvalid = bool(safename) if ALLOWED_EXTRA else (safename in REQUIRED_FILES)
        else:               isvalid = bool(safename) #(not ALLOWED_EXTENSIONS)
    return isvalid, safename

def VALIDATE_FILENAME_SUBMIT(filename): 
    if '.' in filename: 
        name, ext = filename.rsplit('.', 1)
        safename = f'{name}.{ext.lower()}'
        isvalid = bool(safename)
    else:               
        name, ext = filename, ''
        safename = f'{name}'
        isvalid = isvalid = bool(safename)
    return isvalid, safename

def str2bytes(size):
    sizes = dict(KB=2**10, MB=2**20, GB=2**30, TB=2**40)
    return int(float(size[:-2])*sizes.get(size[-2:].upper(), 0))
MAX_UPLOAD_SIZE = str2bytes(args.maxupsize)     # maximum upload file size 
MAX_UPLOAD_COUNT = ( inf if args.maxupcount<0 else args.maxupcount )       # maximum number of files that can be uploaded by one user
INITIAL_UPLOAD_STATUS = []           # a list of notes to be displayed to the users about uploading files
if REQUIRED_FILES: INITIAL_UPLOAD_STATUS.append((-1, f'accepted files [{len(REQUIRED_FILES)}]: {REQUIRED_FILES}'))
INITIAL_UPLOAD_STATUS.append((-1, f'max upload size: {DISPLAY_SIZE_READABLE(MAX_UPLOAD_SIZE)}'))
if not (MAX_UPLOAD_COUNT is inf): INITIAL_UPLOAD_STATUS.append((-1, f'max upload count: {MAX_UPLOAD_COUNT}'))
sprint(f'⚙ Upload Settings ({len(INITIAL_UPLOAD_STATUS)})')
for s in INITIAL_UPLOAD_STATUS: sprint(f' ⇒ {s[1]}')
# ------------------------------------------------------------------------------------------
































def TEMPLATES(style):

    # ******************************************************************************************
    HTML_TEMPLATES = dict(
    # ******************************************************************************************
    board="""""",
    # ******************************************************************************************
    evaluate = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_eval}'+""" {{ config.topic }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">  
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            <a href="{{ url_for('route_eval') }}" class="btn_refresh">Refresh</a>
            <a href="{{ url_for('route_storeuser') }}" class="btn_store">User-Store</a>
            <a href="{{ url_for('route_generate_submit_report') }}" target="_blank" class="btn_board">User-Report</a>
            <button class="btn_purge_large" onclick="confirm_repass()">"""+'Reset Password' + """</button>
                <script>
                    function confirm_repass() {
                    let res = prompt("Enter UID", ""); 
                    if (res != null) {
                        location.href = "{{ url_for('route_repassx',req_uid='::::') }}".replace("::::", res);
                        }
                    }
                </script>
            </div>
            <br>
            {% if success %}
            <span class="admin_mid" style="animation-name: fader_admin_success;">✓ {{ status }} </span>
            {% else %}
            <span class="admin_mid" style="animation-name: fader_admin_failed;">✗ {{ status }} </span>
            {% endif %}
            <br>
            <br>
            <form action="{{ url_for('route_eval') }}" method="post">
                
                    <input id="uid" name="uid" type="text" placeholder="uid" class="txt_submit"/>
                    <br>
                    <br>
                    <input id="score" name="score" type="text" placeholder="score" class="txt_submit"/> 
                    <br>
                    <br>
                    <input id="remark" name="remark" type="text" placeholder="remarks" class="txt_submit"/>
                    <br>
                    <br>
                    <input type="submit" class="btn_submit" value="Submit Evaluation"> 
                    <br>   
                    <br> 
            </form>
            
            <form method='POST' enctype='multipart/form-data'>
                {{form.hidden_tag()}}
                {{form.file()}}
                {{form.submit()}}
            </form>
            <a href="{{ url_for('route_generate_eval_template') }}" class="btn_black">Get CSV-Template</a>
            <br>
        
        </div>
        
        {% if results %}
        <div class="status">
        <table>
        {% for (ruid,rmsg,rstatus) in results %}
            {% if rstatus %}
                <tr class="btn_disablel">
            {% else %}
                <tr class="btn_enablel">
            {% endif %}
                <td>{{ ruid }} ~ </td>
                <td>{{ rmsg }}</td>
                </tr>
        {% endfor %}
        </table>
        </div>
        {% endif %}
                    
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,

    # ******************************************************************************************
    login = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_login}'+""" {{ config.topic }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">  
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">

        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->

        <div align="center">
            <br>
            <div class="topic">{{ config.topic }}</div>
            <br>
            <br>
            <form action="{{ url_for('route_login') }}" method="post">
                <br>
                <div style="font-size: x-large;">{{ warn }}</div>
                <br>
                <div class="msg_login">{{ msg }}</div>
                <br>
                <input id="uid" name="uid" type="text" placeholder="... user-id ..." class="txt_login"/>
                <br>
                <br>
                <input id="passwd" name="passwd" type="password" placeholder="... password ..." class="txt_login"/>
                <br>
                <br>
                {% if config.rename>0 %}
                <input id="named" name="named" type="text" placeholder="... update-name ..." class="txt_login"/>
                <br>
                {% endif %}
                <br>
                <input type="submit" class="btn_login" value=""" +f'"{style.login_}"'+ """> 
                <br>
                <br>
            </form>
        </div>

        <!-- ---------------------------------------------------------->
        
        <div align="center">
        <div>
        <a href="https://github.com/auto-notify-ps/known" target="_blank"><span style="font-size: xx-large;">{{ config.emoji }}</span></a>
        <br>
        {% if config.reg %}
        <a href="{{ url_for('route_new') }}" class="btn_board">""" + f'{style.new_}' +"""</a>
        {% endif %}
        </div>
        <br>
        </div>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    # ******************************************************************************************
    new = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_new}'+""" {{ config.topic }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">  
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">

        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->

        <div align="center">
            <br>
            <div class="topic">{{ config.topic }}</div>
            <br>
            <br>
            <form action="{{ url_for('route_new') }}" method="post">
                <br>
                <div style="font-size: x-large;">{{ warn }}</div>
                <br>
                <div class="msg_login">{{ msg }}</div>
                <br>
                <input id="uid" name="uid" type="text" placeholder="... user-id ..." class="txt_login"/>
                <br>
                <br>
                <input id="passwd" name="passwd" type="password" placeholder="... password ..." class="txt_login"/>
                <br>
                <br>
                <input id="named" name="named" type="text" placeholder="... name ..." class="txt_login"/>
                <br>
                <br>
                <input type="submit" class="btn_board" value=""" + f'"{style.new_}"' +"""> 
                <br>
                <br>
                
            </form>
        </div>

        <!-- ---------------------------------------------------------->
        
        <div align="center">
        <div>
        <span style="font-size: xx-large;">{{ config.emoji }}</span>
        <br>
        <a href="{{ url_for('route_login') }}" class="btn_login">""" + f'{style.login_}' +"""</a>
        
        </div>
        <br>
        </div>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    # ******************************************************************************************
    downloads = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_downloads}'+""" {{ config.topic }} | {{ session.uid }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">           
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">

        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            </div>
            <br>
            <div class="files_status">"""+f'{style.downloads_}'+"""</div>
            <br>
            <div class="files_list_down">
                <ol>
                {% for file in config.dfl %}
                <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}"" >{{ file }}</a></li>
                <br>
                {% endfor %}
                </ol>
            </div>
            <br>
        </div>

        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    # ******************************************************************************************
    storeuser = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_store}'+""" {{ config.topic }} | {{ session.uid }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">   
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
            
        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            <a href="{{ url_for('route_eval') }}" class="btn_submit">"""+f'{style.eval_}'+"""</a>
            {% if not subpath %}
            {% if session.hidden_storeuser %}
                <a href="{{ url_for('route_hidden_show', user_enable='10') }}" class="btn_disable">"""+f'{style.icon_hidden}'+"""</a>
            {% else %}
                <a href="{{ url_for('route_hidden_show', user_enable='11') }}" class="btn_enable">"""+f'{style.icon_hidden}'+"""</a>
            {% endif %}
            {% endif %}
            </div>
            <br>
            <hr>
            <!-- Breadcrumb for navigation -->
            <div class="files_status"> Path: 
                {% if subpath %}
                    <a href="{{ url_for('route_storeuser') }}" class="btn_store">{{ config.storeusername }}</a>{% for part in subpath.split('/') %}🔹<a href="{{ url_for('route_storeuser', subpath='/'.join(subpath.split('/')[:loop.index])) }}" class="btn_store">{{ part }}</a>{% endfor %}  
                {% else %}
                    <a href="{{ url_for('route_storeuser') }}" class="btn_store">{{ config.storeusername }}</a>
                {% endif %}
            </div>
            <hr>
            <!-- Directory Listing -->
            
            <div class="files_list_up">
                <p class="files_status">Folders</p>
                {% for (dir,hdir) in dirs %}
                    {% if (session.hidden_storeuser) or (not hdir) %}
                        <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + dir) }}" class="btn_folder">{{ dir }}</a>
                    {% endif %}
                {% endfor %}
            </div>
            <hr>
            
            <div class="files_list_down">
                <p class="files_status">Files</p>
                <ol>
                {% for (file, hfile) in files %}
                {% if (session.hidden_storeuser) or (not hfile) %}
                    <li>
                    <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + file, get='') }}">"""+f'{style.icon_getfile}'+"""</a> 
                    <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + file) }}" target="_blank">{{ file }}</a>
                    {% if file.lower().endswith('.ipynb') %}
                    <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + file, html='') }}">"""+f'{style.icon_gethtml}'+"""</a> 
                    {% endif %}
                    </li>
                {% endif %}
                
                {% endfor %}
                </ol>
            </div>
            <br>
        </div>

        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    store = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_store}'+""" {{ config.topic }} | {{ session.uid }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">      
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
        
        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            {% if not subpath %}
            {% if session.hidden_store %}
                <a href="{{ url_for('route_hidden_show', user_enable='00') }}" class="btn_disable">"""+f'{style.icon_hidden}'+"""</a>
            {% else %}
                <a href="{{ url_for('route_hidden_show', user_enable='01') }}" class="btn_enable">"""+f'{style.icon_hidden}'+"""</a>
            {% endif %}
            {% endif %}
            {% if "X" in session.admind or "+" in session.admind %}
            <form method='POST' enctype='multipart/form-data'>
                {{form.hidden_tag()}}
                {{form.file()}}
                {{form.submit()}}
            </form>
            {% endif %}
            </div>
            <br>
            <hr>
            <!-- Breadcrumb for navigation -->
            <div class="files_status"> Path: 
                {% if subpath %}
                    <a href="{{ url_for('route_store') }}" class="btn_store">{{ config.storename }}</a>{% for part in subpath.split('/') %}🔹<a href="{{ url_for('route_store', subpath='/'.join(subpath.split('/')[:loop.index])) }}" class="btn_store">{{ part }}</a>{% endfor %}  
                {% else %}
                    <a href="{{ url_for('route_store') }}" class="btn_store">{{ config.storename }}</a>
                {% endif %}
            </div>
            <hr>
            <!-- Directory Listing -->
            
            <div class="files_list_up">
                <p class="files_status">Folders</p>
                {% for (dir,hdir) in dirs %}
                    {% if (session.hidden_store) or (not hdir) %}
                        <a href="{{ url_for('route_store', subpath=subpath + '/' + dir) }}" class="btn_folder">{{ dir }}</a>
                    {% endif %}
                {% endfor %}
            </div>
            <hr>
            
            <div class="files_list_down">
                <p class="files_status">Files</p>
                <ol>
                {% for (file, hfile) in files %}
                {% if (session.hidden_store) or (not hfile) %}
                    <li>
                    <a href="{{ url_for('route_store', subpath=subpath + '/' + file, get='') }}">"""+f'{style.icon_getfile}'+"""</a> 
                    <a href="{{ url_for('route_store', subpath=subpath + '/' + file) }}" target="_blank" >{{ file }}</a>
                
                    </li>
                {% endif %}
                
                {% endfor %}
                </ol>
            </div>
            <br>
        </div>

        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    # ******************************************************************************************
    uploads = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_uploads}'+""" {{ config.topic }} | {{ session.uid }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">        
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
    
        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            </div>
            <br>
            <div class="files_status">"""+f'{style.uploads_}'+"""</div>
            <br>
            <div class="files_list_down">
                <ol>
                {% for file in session.filed %}
                <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}">{{ file }}</a></li>
                <br>
                {% endfor %}
                </ol>
            </div>
            <br>
        </div>

        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    # ******************************************************************************************
    reports = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_report}'+""" {{ config.topic }} | {{ session.uid }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">     
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
        
        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            </div>
            <br>
            <div class="files_status">"""+f'{style.report_}'+"""</div>
            <br>
            <div class="files_list_down">
                <ol>
                {% for file in session.reported %}
                <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}"  target="_blank">{{ file }}</a></li>
                <br>
                {% endfor %}
                </ol>
            </div>
            <br>
        </div>

        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    # ******************************************************************************************
    home="""
    <html>
        <head>
            <meta charset="UTF-8">
            <title> """+f'{style.icon_home}'+""" {{ config.topic }} | {{ session.uid }} </title>
            <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">			
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
            
        </head>
        <body>
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        
        <div align="left" style="padding: 20px;">
            <div class="topic_mid">{{ config.topic }}</div>
            <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            {% if "S" in session.admind %}
            <a href="{{ url_for('route_uploads') }}" class="btn_upload">"""+f'{style.uploads_}'+"""</a>
            {% endif %}
            {% if "D" in session.admind %}
            <a href="{{ url_for('route_downloads') }}" class="btn_download">"""+f'{style.downloads_}'+"""</a>
            {% endif %}
            {% if "A" in session.admind %}
            <a href="{{ url_for('route_store') }}" class="btn_store">"""+f'{style.store_}'+"""</a>
            {% endif %}
            {% if "B" in session.admind and config.board %}
            <a href="{{ url_for('route_board') }}" class="btn_board" target="_blank">"""+f'{style.board_}'+"""</a>
            {% endif %}
            {% if 'X' in session.admind or '+' in session.admind %}
            <a href="{{ url_for('route_eval') }}" class="btn_submit">"""+f'{style.eval_}'+"""</a>
            {% endif %}
            {% if 'R' in session.admind %}
            <a href="{{ url_for('route_reports') }}" class="btn_report">"""+f'{style.report_}'+"""</a>
            {% endif %}
            
            </div>
            <br>
            {% if "U" in session.admind %}
                <div class="status">
                    <ol>
                    {% for s,f in status %}
                    {% if s %}
                    {% if s<0 %}
                    <li style="color: """+f'{style.item_normal}'+""";">{{ f }}</li>
                    {% else %}
                    <li style="color: """+f'{style.item_true}'+""";">{{ f }}</li>
                    {% endif %}
                    {% else %}
                    <li style="color: """+f'{style.item_false}'+""";">{{ f }}</li>
                    {% endif %}
                    {% endfor %}
                    </ol>
                </div>
                <br>
                {% if submitted<1 %}
                    {% if config.muc!=0 %}
                    <form method='POST' enctype='multipart/form-data'>
                        {{form.hidden_tag()}}
                        {{form.file()}}
                        {{form.submit()}}
                    </form>
                    {% endif %}
                {% else %}
                    <div class="upword">Your Score is <span style="color:seagreen;">{{ score }}</span>  </div>
                {% endif %}
                <br>
                    
                <div> <span class="upword">Uploads</span> 
                    
                {% if submitted<1 and config.muc!=0 %}
                    <a href="{{ url_for('route_uploadf') }}" class="btn_refresh_small">Refresh</a>
                    <button class="btn_purge" onclick="confirm_purge()">Purge</button>
                    <script>
                        function confirm_purge() {
                        let res = confirm("Purge all the uploaded files now?");
                        if (res == true) {
                            location.href = "{{ url_for('route_purge') }}";
                            }
                        }
                    </script>
                {% endif %}
                </div>
                <br>

                <div class="files_list_up">
                    <ol>
                    {% for f in session.filed %}
                        <li>{{ f }}</li>
                    {% endfor %}
                    </ol>
                </div>
            {% endif %}
            
                
        <!-- ---------------------------------------------------------->
        </br>
        <!-- ---------------------------------------------------------->
        </body>
    </html>
    """,
    #******************************************************************************************

    # ******************************************************************************************
    )
    # ******************************************************************************************
    CSS_TEMPLATES = dict(
    # ****************************************************************************************** 
    style = f""" 

    body {{
        background-color: {style.bgcolor};
        color: {style.fgcolor};
    }}

    a {{
        color: {style.refcolor};
        text-decoration: none;
    }}

    .files_list_up{{
        padding: 10px 10px;
        background-color: {style.flup_bgcolor}; 
        color: {style.flup_fgcolor};
        font-size: medium;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }}

    .files_list_down{{
        padding: 10px 10px;
        background-color: {style.fldown_bgcolor}; 
        color: {style.fldown_fgcolor};
        font-size: large;
        font-weight: bold;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }}

    .topic{{
        color:{style.fgcolor};
        font-size: xxx-large;
        font-weight: bold;
        font-family:monospace;    
    }}

    .msg_login{{
        color: {style.msgcolor}; 
        font-size: large;
        font-weight: bold;
        font-family:monospace;    
        animation-duration: 3s; 
        animation-name: fader_msg;
    }}
    @keyframes fader_msg {{from {{color: {style.bgcolor};}} to {{color: {style.msgcolor}; }} }}



    .topic_mid{{
        color: {style.fgcolor};
        font-size: x-large;
        font-style: italic;
        font-weight: bold;
        font-family:monospace;    
    }}

    .userword{{
        color: {style.fgcolor};
        font-weight: bold;
        font-family:monospace;    
        font-size: xxx-large;
    }}


    .upword{{
        color: {style.fgcolor};
        font-weight: bold;
        font-family:monospace;    
        font-size: xx-large;

    }}

    .status{{
        padding: 10px 10px;
        background-color: {style.item_bgcolor}; 
        color: {style.item_normal};
        font-size: medium;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }}


    .files_status{{
        font-weight: bold;
        font-size: x-large;
        font-family:monospace;
    }}


    .admin_mid{{
        color: {style.fgcolor}; 
        font-size: x-large;
        font-weight: bold;
        font-family:monospace;    
        animation-duration: 10s;
    }}
    @keyframes fader_admin_failed {{from {{color: {style.item_false};}} to {{color: {style.fgcolor}; }} }}
    @keyframes fader_admin_success {{from {{color: {style.item_true};}} to {{color: {style.fgcolor}; }} }}
    @keyframes fader_admin_normal {{from {{color: {style.item_normal};}} to {{color: {style.fgcolor}; }} }}



    .btn_enablel {{
        padding: 2px 10px 2px;
        color: {style.item_false}; 
        font-size: medium;
        border-radius: 2px;
        font-family:monospace;
        text-decoration: none;
    }}


    .btn_disablel {{
        padding: 2px 10px 2px;
        color: {style.item_true}; 
        font-size: medium;
        border-radius: 2px;
        font-family:monospace;
        text-decoration: none;
    }}

    .btn_enable {{
        padding: 2px 10px 2px;
        background-color: {style.item_false}; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }}


    .btn_disable {{
        padding: 2px 10px 2px;
        background-color: {style.item_true}; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }}

    """ + """

    #file {
        border-style: solid;
        border-radius: 10px;
        font-family:monospace;
        background-color: #232323;
        border-color: #232323;
        color: #FFFFFF;
        font-size: small;
    }
    #submit {
        padding: 2px 10px 2px;
        background-color: #232323; 
        color: #FFFFFF;
        font-family:monospace;
        font-weight: bold;
        font-size: large;
        border-style: solid;
        border-radius: 10px;
        border-color: #232323;
        text-decoration: none;
        font-size: small;
    }
    #submit:hover {
    box-shadow: 0 12px 16px 0 rgba(0, 0, 0,0.24), 0 17px 50px 0 rgba(0, 0, 0,0.19);
    }



    .bridge{
        line-height: 2;
    }



    .txt_submit{

        text-align: left;
        font-family:monospace;
        border: 1px;
        background: rgb(218, 187, 255);
        appearance: none;
        position: relative;
        border-radius: 3px;
        padding: 5px 5px 5px 5px;
        line-height: 1.5;
        color: #8225c2;
        font-size: 16px;
        font-weight: 350;
        height: 24px;
    }
    ::placeholder {
        color: #8225c2;
        opacity: 1;
        font-family:monospace;   
    }

    .txt_login{

        text-align: center;
        font-family:monospace;

        box-shadow: inset #abacaf 0 0 0 2px;
        border: 0;
        background: rgba(0, 0, 0, 0);
        appearance: none;
        position: relative;
        border-radius: 3px;
        padding: 9px 12px;
        line-height: 1.4;
        color: rgb(0, 0, 0);
        font-size: 16px;
        font-weight: 400;
        height: 40px;
        transition: all .2s ease;
        :hover{
            box-shadow: 0 0 0 0 #fff inset, #1de9b6 0 0 0 2px;
        }
        :focus{
            background: #fff;
            outline: 0;
            box-shadow: 0 0 0 0 #fff inset, #1de9b6 0 0 0 3px;
        }
    }
    ::placeholder {
        color: #888686;
        opacity: 1;
        font-weight: bold;
        font-style: oblique;
        font-family:monospace;   
    }


    .txt_login_small{
        box-shadow: inset #abacaf 0 0 0 2px;
        text-align: center;
        font-family:monospace;
        border: 0;
        background: rgba(0, 0, 0, 0);
        appearance: none;
        position: absolute;
        border-radius: 3px;
        padding: 9px 12px;
        margin: 0px 0px 0px 4px;
        line-height: 1.4;
        color: rgb(0, 0, 0);
        font-size: 16px;
        font-weight: 400;
        height: 40px;
        width: 45px;
        transition: all .2s ease;
        :hover{
            box-shadow: 0 0 0 0 #fff inset, #1de9b6 0 0 0 2px;
        }
        :focus{
            background: #fff;
            outline: 0;
            box-shadow: 0 0 0 0 #fff inset, #1de9b6 0 0 0 3px;
        }
    }




    .btn_logout {
        padding: 2px 10px 2px;
        background-color: #060472; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }


    .btn_refresh_small {
        padding: 2px 10px 2px;
        background-color: #6daa43; 
        color: #FFFFFF;
        font-size: small;
        border-style: none;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_refresh {
        padding: 2px 10px 2px;
        background-color: #6daa43; 
        color: #FFFFFF;
        font-size: large;
        font-weight: bold;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_purge {
        padding: 2px 10px 2px;
        background-color: #9a0808; 
        border-style: none;
        color: #FFFFFF;
        font-size: small;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_purge_large {
        padding: 2px 10px 2px;
        background-color: #9a0808; 
        border-style: none;
        color: #FFFFFF;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_submit {
        padding: 2px 10px 2px;
        background-color: #8225c2; 
        border-style: none;
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_report {
        padding: 2px 10px 2px;
        background-color: #c23f79; 
        border-style: none;
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }
    .btn_black {
        padding: 2px 10px 2px;
        background-color: #2b2b2b; 
        border-style: none;
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_store_actions {
        padding: 2px 2px 2px 2px;
        background-color: #FFFFFF; 
        border-style: solid;
        border-width: thin;
        border-color: #000000;
        color: #000000;
        font-weight: bold;
        font-size: medium;
        border-radius: 5px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_folder {
        padding: 2px 10px 2px;
        background-color: #934343; 
        border-style: none;
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
        line-height: 2;
    }

    .btn_board {
        padding: 2px 10px 2px;
        background-color: #934377; 
        border-style: none;
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }


    .btn_login {
        padding: 2px 10px 2px;
        background-color: #060472; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
        border-style:  none;
    }

    .btn_download {
        padding: 2px 10px 2px;
        background-color: #089a28; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_store{
        padding: 2px 10px 2px;
        background-color: #10a58a; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_upload {
        padding: 2px 10px 2px;
        background-color: #0b7daa; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }

    .btn_home {
        padding: 2px 10px 2px;
        background-color: #a19636; 
        color: #FFFFFF;
        font-weight: bold;
        font-size: large;
        border-radius: 10px;
        font-family:monospace;
        text-decoration: none;
    }




    """
    )
    # ******************************************************************************************
    return HTML_TEMPLATES, CSS_TEMPLATES
    # ****************************************************************************************** 


# ------------------------------------------------------------------------------------------
# html pages
# ------------------------------------------------------------------------------------------
style = Fake(**args.style)
HTML_TEMPLATES, CSS_TEMPLATES = TEMPLATES(style)
# ------------------------------------------------------------------------------------------
for k,v in HTML_TEMPLATES.items():
    h = os.path.join(HTMLDIR, f"{k}.html")
    if (not os.path.isfile(h)) or bool(parsed.cos):
        try:
            with open(h, 'w', encoding='utf-8') as f: f.write(v)
        except: fexit(f'[!] Cannot create html "{k}" at {h}')
# ------------------------------------------------------------------------------------------
for k,v in CSS_TEMPLATES.items():
    h = os.path.join(HTMLDIR, f"{k}.css")
    if (not os.path.isfile(h)) or bool(parsed.cos):
        try:
            with open(h, 'w', encoding='utf-8') as f: f.write(v)
        except: fexit(f'[!] Cannot create css "{k}" at {h}')
# ------------------------------------------------------------------------------------------
sprint(f'↪ Created html/css templates @ {HTMLDIR}')
# ------------------------------------------------------------------------------------------
favicon_path = os.path.join(HTMLDIR, f"favicon.ico")
if not os.path.exists(favicon_path):
    try:
        with open( favicon_path, 'wb') as f: f.write((b''.join([i.to_bytes() for i in [
    137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,32,0,0,0,32,8,6,0,0,1,4,125,74,98,0,0,0,9,112,72,89,
    115,0,0,13,214,0,0,13,214,1,144,111,121,156,0,0,9,130,73,68,65,84,88,9,205,87,121,84,148,215,21,255,125,223,236,48,44,195,190,202,
    166,17,84,20,149,40,46,71,69,196,163,104,84,2,99,92,18,143,113,1,219,186,212,86,49,61,193,35,86,115,76,76,172,75,244,52,154,26,27,143,
    165,169,40,42,46,168,32,166,41,46,36,106,100,81,81,65,20,144,69,97,128,97,96,54,102,190,222,247,17,60,68,123,98,19,251,71,223,57,223,188,
    247,238,187,247,190,123,223,93,135,19,4,1,63,26,147,39,107,15,9,159,105,69,232,142,37,90,129,99,167,191,155,163,21,74,155,129,37,3,0,158,
    1,18,135,58,224,252,76,204,59,89,195,131,123,158,7,183,121,65,178,96,244,234,235,125,241,122,69,156,74,194,101,242,183,90,57,8,117,149,141,137,
    33,200,180,1,115,248,167,70,148,172,91,18,12,169,143,11,36,28,46,128,174,77,160,239,239,140,249,111,82,87,250,253,48,15,101,51,193,191,21,197,
    152,53,227,45,33,115,77,16,150,125,242,0,106,31,95,157,167,131,193,109,109,162,39,18,50,30,64,68,16,206,164,8,48,11,248,254,161,21,31,93,
    225,17,166,54,34,58,64,138,71,45,246,23,197,100,172,123,143,23,244,232,125,200,214,28,9,82,66,243,148,243,231,179,234,122,31,18,124,41,237,181,
    82,171,13,145,9,65,120,236,52,77,107,246,9,240,125,157,128,185,139,189,235,253,219,173,192,7,215,233,237,56,18,115,209,252,96,124,152,18,168,168,
    174,107,94,241,184,186,206,63,116,66,48,52,81,254,32,75,100,136,90,228,175,211,10,5,53,2,42,244,28,252,213,64,113,19,144,66,47,63,103,87,
    22,39,29,57,46,57,253,111,15,120,4,186,240,24,27,226,128,226,7,70,228,46,117,128,201,218,109,72,190,232,155,35,155,83,98,84,80,73,129,141,
    39,90,250,234,141,246,78,185,146,195,145,98,179,40,243,203,213,124,222,158,189,85,253,111,214,236,157,26,9,209,203,70,34,49,141,104,46,185,152,159,
    53,132,12,91,179,103,239,174,64,198,132,214,131,105,106,43,171,168,247,151,241,184,196,19,34,211,128,225,115,19,226,146,71,31,91,224,114,201,213,149,
    199,173,234,46,252,225,140,1,31,143,6,126,91,136,218,190,33,126,86,146,112,74,125,147,62,40,45,188,227,124,246,3,192,199,67,142,149,99,85,144,
    168,56,204,218,219,102,23,159,145,164,144,218,5,172,148,66,216,54,187,47,240,117,163,4,73,147,60,240,233,177,39,226,45,51,71,185,162,86,103,133,
    188,197,0,165,132,44,85,45,220,81,201,249,21,100,221,11,162,147,19,151,137,114,30,219,166,71,42,33,168,149,248,242,29,103,60,190,223,10,57,33,
    59,209,254,82,137,30,91,98,101,216,152,164,65,133,81,138,190,30,210,8,82,53,159,52,232,142,18,186,230,126,27,185,99,106,140,146,193,96,178,8,
    40,215,243,48,219,81,190,106,16,254,146,62,207,23,167,74,77,248,213,225,118,236,159,235,132,112,47,9,147,172,157,225,138,42,176,5,27,147,226,181,
    57,179,6,41,222,184,219,198,161,170,193,92,127,234,204,97,49,66,198,142,79,10,94,63,215,167,234,224,185,167,232,32,31,235,180,8,33,36,254,67,
    70,243,82,59,51,164,159,26,175,204,64,124,196,159,186,225,101,103,204,145,206,19,146,43,125,159,147,94,159,51,2,114,156,97,228,68,55,216,154,13,
    150,45,104,47,6,36,225,179,119,217,70,223,64,250,222,99,18,196,147,87,177,64,220,71,143,40,140,143,211,146,27,161,153,136,34,105,238,25,223,178,
    69,236,36,109,49,249,203,99,50,225,28,218,178,243,3,162,10,118,59,240,254,112,96,122,16,176,125,12,46,125,127,175,142,49,201,165,143,221,30,233,
    234,228,48,52,105,154,214,84,144,140,193,83,200,185,99,125,5,80,38,96,35,143,155,57,109,182,237,120,170,11,111,51,10,216,85,104,68,67,147,5,
    111,134,2,91,203,29,227,125,61,156,171,57,142,59,91,81,85,39,219,49,22,1,107,47,3,91,18,212,24,216,71,138,214,86,59,18,15,182,141,225,
    226,227,181,25,230,46,97,195,238,101,1,8,245,145,163,139,228,219,158,211,132,226,187,237,240,11,244,109,169,169,215,29,209,112,230,165,235,23,247,129,
    191,70,10,35,37,130,197,59,171,161,239,180,163,32,63,139,101,28,49,133,198,25,45,194,167,83,251,32,194,68,162,89,52,106,4,184,201,112,226,74,
    171,24,117,43,18,189,112,52,191,9,19,188,109,56,92,1,116,129,251,61,69,228,46,122,244,46,241,13,44,118,120,46,31,194,69,132,105,56,20,54,
    114,240,232,52,160,242,94,43,227,13,55,5,80,84,248,4,1,74,27,202,90,56,172,25,198,193,83,133,109,140,152,157,83,184,0,97,97,3,75,159,
    90,120,84,24,120,100,47,114,193,232,126,42,124,121,195,2,137,66,65,110,219,133,232,96,5,214,37,56,161,86,47,160,193,42,69,89,131,13,161,97,
    3,175,84,86,222,174,20,37,152,30,46,199,142,89,106,188,17,33,131,133,116,124,255,180,1,14,42,9,100,50,25,166,70,59,35,134,30,109,3,193,
    22,143,86,225,122,173,21,167,87,105,208,106,18,152,233,65,169,16,56,93,110,105,183,218,5,167,181,113,142,98,196,77,11,151,33,116,128,59,154,74,
    155,247,255,177,200,52,230,126,160,50,252,195,120,71,124,113,217,136,125,90,39,140,221,170,131,171,138,203,100,180,162,4,228,72,161,238,14,60,222,57,
    164,199,159,41,92,43,44,114,100,100,54,116,142,243,54,21,220,172,179,77,223,52,223,7,59,191,238,192,107,30,18,92,171,233,130,74,198,149,156,59,
    151,85,197,24,60,11,103,114,209,96,7,57,87,229,40,3,22,76,246,196,166,175,26,66,10,255,121,244,33,67,154,158,48,187,46,196,71,225,219,223,
    69,192,241,50,243,201,252,188,172,25,12,206,198,255,65,52,190,106,93,232,86,228,151,255,138,70,248,229,228,175,78,249,172,176,253,7,86,13,4,99,
    249,105,59,121,125,11,59,167,204,240,49,77,1,148,155,230,178,253,243,131,206,191,34,88,53,157,167,177,51,114,12,13,77,171,233,99,205,132,15,125,
    207,143,39,76,128,171,4,29,73,174,8,42,77,88,77,53,52,196,5,216,124,13,24,238,1,4,57,147,20,183,209,166,51,35,181,176,32,235,31,116,
    201,78,66,245,164,75,230,245,230,70,112,214,106,61,33,248,170,177,19,181,111,81,8,239,93,58,0,46,15,245,192,13,106,55,210,163,129,170,54,210,
    134,250,29,10,253,30,247,45,146,80,24,187,216,236,152,60,33,76,134,133,35,84,168,54,73,144,243,16,104,234,16,112,91,7,76,165,252,183,54,26,
    202,81,94,72,126,236,48,48,173,188,182,53,189,143,143,198,39,122,248,136,133,223,93,47,202,102,66,208,229,153,18,158,175,41,42,171,206,25,49,40,
    242,206,142,49,152,187,34,10,74,93,39,176,191,156,16,40,103,221,235,148,66,237,42,199,172,72,5,236,148,176,30,232,236,212,61,97,59,23,51,62,
    217,81,163,226,110,46,139,81,245,149,208,11,176,64,171,211,219,33,16,145,132,50,141,158,234,109,43,229,90,144,148,211,3,5,216,9,158,93,41,156,
    13,15,243,173,226,121,78,205,216,219,237,66,123,121,101,125,72,82,24,55,133,226,10,167,106,40,60,137,25,69,59,156,21,28,108,116,33,131,251,57,
    243,136,166,180,194,246,159,93,53,86,180,24,133,168,158,116,74,143,142,131,132,51,67,37,231,196,252,163,32,123,80,138,133,92,198,33,192,93,142,33,
    33,74,248,122,202,112,250,106,27,246,229,54,179,226,111,146,203,152,200,244,164,86,155,93,41,231,148,41,83,221,49,45,198,5,245,79,173,40,174,50,
    161,182,217,34,230,54,198,211,76,239,158,123,77,47,242,36,29,114,136,108,1,249,22,117,0,207,141,137,147,180,105,86,59,50,2,212,80,69,145,15,
    244,33,29,163,220,1,15,21,112,189,81,192,181,39,84,240,172,28,234,58,0,157,169,155,216,141,26,18,63,71,192,93,38,32,218,139,124,199,155,67,
    147,17,184,217,76,30,105,160,153,124,160,214,0,35,117,102,25,84,68,182,246,190,242,153,0,228,140,78,102,27,190,75,29,128,254,163,124,1,214,179,
    150,234,136,145,153,131,143,139,4,254,212,189,5,208,236,229,196,227,236,29,11,174,62,178,66,202,51,215,165,2,67,61,90,76,144,12,83,34,228,120,
    210,110,71,109,155,13,143,169,228,53,208,236,161,16,16,233,38,96,98,32,135,43,245,192,222,219,184,171,144,224,117,210,94,108,141,196,108,222,205,4,
    169,222,142,92,127,141,151,18,158,161,114,140,215,216,112,49,175,19,159,204,112,132,55,245,199,102,106,137,62,42,232,68,209,35,1,228,51,232,231,43,
    135,141,119,16,5,144,216,59,161,150,11,56,89,102,134,19,217,252,189,137,14,100,58,160,177,205,142,53,167,58,48,126,176,3,60,53,18,184,218,44,
    240,126,100,234,223,212,41,164,16,33,107,45,126,104,238,104,65,237,66,35,79,22,109,37,167,75,59,97,192,137,219,22,28,156,231,4,79,210,120,11,
    9,178,62,183,3,105,19,84,228,139,2,250,245,115,134,171,154,239,50,152,113,183,221,140,114,13,173,25,140,157,49,156,116,194,101,52,140,150,241,56,
    113,203,34,242,100,13,36,187,131,221,197,46,103,227,153,9,216,134,250,154,61,84,106,126,125,96,182,26,238,110,18,92,46,183,96,55,213,176,63,205,
    84,83,171,202,97,249,49,3,54,188,237,135,139,37,6,236,202,109,222,124,251,93,9,121,6,132,1,7,108,166,149,83,221,211,99,7,171,177,241,80,
    29,118,39,170,97,233,18,176,58,199,128,21,84,68,71,83,193,109,214,217,240,238,97,114,4,171,176,135,170,209,114,118,31,27,63,18,128,1,168,75,
    137,232,178,227,56,53,39,175,69,249,74,176,115,182,19,110,84,90,177,179,200,130,189,203,3,113,168,64,135,191,94,208,125,113,233,155,163,139,25,126,
    207,24,51,46,105,255,194,73,110,139,222,142,117,67,234,238,26,172,26,41,199,48,202,45,171,168,37,190,89,111,99,127,14,239,73,121,204,202,203,203,
    186,211,67,195,230,23,4,232,57,156,255,166,150,171,215,99,50,181,76,137,74,41,23,73,73,195,153,90,38,29,81,164,95,188,112,228,95,61,120,189,
    231,216,184,228,113,212,6,109,146,74,56,55,122,102,189,169,75,40,85,72,185,108,63,103,228,29,202,206,234,246,216,222,4,180,126,229,122,252,28,191,
    159,189,21,19,201,207,166,250,31,18,252,27,83,9,228,212,162,170,157,114,0,0,0,0,73,69,78,68,174,66,96,130,
    ]
    ])))         
    except: pass
# ------------------------------------------------------------------------------------------
# delete pages dict after creation? #- keep the keys to "coe"
HTML_TEMPLATES_KEYS = tuple(HTML_TEMPLATES.keys()) #{k:None for k in HTML_TEMPLATES} 
CSS_TEMPLATES_KEYS = tuple(CSS_TEMPLATES.keys()) #{k:None for k in CSS_TEMPLATES}
del HTML_TEMPLATES, CSS_TEMPLATES_KEYS
# ------------------------------------------------------------------------------------------
# Board
# ------------------------------------------------------------------------------------------
BOARD_FILE_MD = None
BOARD_PAGE = ""
if args.board:
    if has_nbconvert_package:
        BOARD_FILE_MD = os.path.join(BASEDIR, f'{args.board}')
        if  os.path.isfile(BOARD_FILE_MD): sprint(f'⚙ Board File: {BOARD_FILE_MD}')
        else: 
            sprint(f'⚙ Board File: {BOARD_FILE_MD} not found - trying to create...')
            try:
                with open(BOARD_FILE_MD, 'w', encoding='utf-8') as f: f.write(NEW_NOTEBOOK_STR(f'# {args.topic}'))
                sprint(f'⚙ Board File: {BOARD_FILE_MD} was created successfully!')
            except:
                BOARD_FILE_MD = None
                sprint(f'⚙ Board File: {BOARD_FILE_MD} could not be created - Board will not be available!')
    else: sprint(f'[!] Board will not be enabled since it requires nbconvert')
if not BOARD_FILE_MD:   sprint(f'⚙ Board: Not Available')
else: sprint(f'⚙ Board: Is Available')
# ------------------------------------------------------------------------------------------
def update_board(): 
    global BOARD_PAGE
    res = False
    if BOARD_FILE_MD:
        try: 
            page,_ = HTMLExporter(template_name=style.template_board).from_file(BOARD_FILE_MD, {'metadata':{'name':f'{style.icon_board} {style.board_} | {args.topic}'}}) 
            BOARD_PAGE = page
            sprint(f'⚙ Board File was updated: {BOARD_FILE_MD}')
            res=True
        except: 
            BOARD_PAGE=""
            sprint(f'⚙ Board File could not be updated: {BOARD_FILE_MD}')
    else: BOARD_PAGE=""
    return res

_ = update_board()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# Database Read/Write
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def read_logindb_from_disk():
    db_frame, res = READ_DB_FROM_DISK(LOGIN_XL_PATH, 1)
    if res: sprint(f'⇒ Loaded login file: {LOGIN_XL_PATH}')
    else: sprint(f'⇒ Failed reading login file: {LOGIN_XL_PATH}')
    return db_frame
def read_evaldb_from_disk():
    dbsub_frame = dict()
    if EVAL_XL_PATH: 
        dbsub_frame, ressub = READ_DB_FROM_DISK(EVAL_XL_PATH, 0)
        if ressub: sprint(f'⇒ Loaded evaluation file: {EVAL_XL_PATH}')
        else: sprint(f'⇒ Did not load evaluation file: [{EVAL_XL_PATH}] exists={os.path.exists(EVAL_XL_PATH)} isfile={os.path.isfile(EVAL_XL_PATH)}')
    return dbsub_frame
# ------------------------------------------------------------------------------------------
def write_logindb_to_disk(db_frame): # will change the order
    res = WRITE_DB_TO_DISK(LOGIN_XL_PATH, db_frame, LOGIN_ORD)
    if res: sprint(f'⇒ Persisted login file: {LOGIN_XL_PATH}')
    else:  sprint(f'⇒ PermissionError - {LOGIN_XL_PATH} might be open, close it first.')
    return res
def write_evaldb_to_disk(dbsub_frame, verbose=True): # will change the order
    ressub = True
    if EVAL_XL_PATH: 
        ressub = WRITE_DB_TO_DISK(EVAL_XL_PATH, dbsub_frame, EVAL_ORD)
        if verbose:
            if ressub: sprint(f'⇒ Persisted evaluation file: {EVAL_XL_PATH}')
            else:  sprint(f'⇒ PermissionError - {EVAL_XL_PATH} might be open, close it first.')
    return ressub
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
db =    read_logindb_from_disk()  #<----------- Created database here 
dbsub = read_evaldb_from_disk()  #<----------- Created database here 
sprint('↷ persisted eval-db [{}]'.format(write_evaldb_to_disk(dbsub)))
dbevalset = set([k for k,v in db.items() if '-' not in v[0]])
# ------------------------------------------------------------------------------------------
# Check user upload requirements
# ------------------------------------------------------------------------------------------
def GetUserFiles(uid): 
    if not REQUIRED_FILES: return True # no files are required to be uploaded
    udir = os.path.join( app.config['uploads'], uid)
    has_udir = os.path.isdir(udir)
    if has_udir: return not (False in [os.path.isfile(os.path.join(udir, f)) for f in REQUIRED_FILES])
    else: return False
class UploadFileForm(FlaskForm): # The upload form using FlaskForm
    file = MultipleFileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")
# ------------------------------------------------------------------------------------------
# application setting and instance
# ------------------------------------------------------------------------------------------
LOGIN_REG_TEXT =        '👤'
LOGIN_NEED_TEXT =       '🔒'
LOGIN_FAIL_TEXT =       '❌'     
LOGIN_NEW_TEXT =        '🔥'
LOGIN_CREATE_TEXT =     '🔑'    
#%% [APP DEFINE] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
app = Flask(
    __name__,
    static_folder=HTMLDIR,      # Set your custom static folder path here
    template_folder=HTMLDIR,   # Set your custom templates folder path here
    instance_relative_config = True,
    instance_path = WORKDIR,
)
# ------------------------------------------------------------------------------------------
# app config
# ------------------------------------------------------------------------------------------
app.secret_key =          APP_SECRET_KEY
app.config['base'] =      BASEDIR
app.config['uploads'] =   UPLOAD_FOLDER_PATH
app.config['reports'] =   REPORT_FOLDER_PATH
app.config['downloads'] = DOWNLOAD_FOLDER_PATH
app.config['store'] =     STORE_FOLDER_PATH
app.config['storename'] =  os.path.basename(STORE_FOLDER_PATH)
app.config['storeuser'] =     UPLOAD_FOLDER_PATH
app.config['storeusername'] =  os.path.basename(UPLOAD_FOLDER_PATH)
app.config['emoji'] =     args.emoji
app.config['topic'] =     args.topic
app.config['dfl'] =       GET_FILE_LIST(DOWNLOAD_FOLDER_PATH)
app.config['rename'] =    int(args.rename)
app.config['muc'] =       MAX_UPLOAD_COUNT
app.config['board'] =     (BOARD_FILE_MD is not None)
app.config['reg'] =       (parsed.reg)
app.config['repass'] =    bool(args.repass)
app.config['eip'] =       bool(parsed.eip)
app.config['apac'] =    f'{parsed.access}'.strip().upper()
# ------------------------------------------------------------------------------------------



#%% [ROUTES] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# ------------------------------------------------------------------------------------------
# login
# ------------------------------------------------------------------------------------------
@app.route('/', methods =['GET', 'POST'])
def route_login():
    if request.method == 'POST' and 'uid' in request.form and 'passwd' in request.form:
        global db
        in_uid = f"{request.form['uid']}"
        in_passwd = f"{request.form['passwd']}"
        in_name = f'{request.form["named"]}' if 'named' in request.form else ''
        in_emoji = app.config['emoji']
        in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
        valid_query, valid_name = VALIDATE_UID(in_query) , VALIDATE_NAME(in_name)
        if not valid_query : record=None
        else: record = db.get(in_query, None)
        if record is not None: 
            admind, uid, named, passwd = record
            if not passwd: # fist login
                if in_passwd: # new password provided
                    if VALIDATE_PASS(in_passwd): # new password is valid
                        db[uid][3]=in_passwd 
                        if in_name!=named and valid_name and (app.config['rename']>0) : 
                            db[uid][2]=in_name
                            dprint(f'⇒ {uid} ◦ {named} updated name to "{in_name}" via {request.remote_addr}') 
                            named = in_name
                        else:
                            if in_name: dprint(f'⇒ {uid} ◦ {named} provided invalid name "{in_name}" (will not update)') 

                        warn = LOGIN_CREATE_TEXT
                        msg = f'[{in_uid}] ({named}) New password was created successfully'
                        dprint(f'● {in_uid} {in_emoji} {named} just joined via {request.remote_addr}')
           
                    else: # new password is invalid valid 
                        warn = LOGIN_NEW_TEXT
                        msg=f'[{in_uid}] New password is invalid - can use any of the alphabets (A-Z, a-z), numbers (0-9), underscore (_), dot (.) and at-symbol (@) only'
                        
                                               
                else: #new password not provided                
                    warn = LOGIN_NEW_TEXT
                    msg = f'[{in_uid}] New password required - can use any of the alphabets (A-Z, a-z), numbers (0-9), underscore (_), dot (.) and at-symbol (@) only'
                                           
            else: # re login
                if in_passwd: # password provided 
                    if in_passwd==passwd:
                        folder_name = os.path.join(app.config['uploads'], uid)
                        folder_report = os.path.join(app.config['reports'], uid) 
                        try:
                            os.makedirs(folder_name, exist_ok=True)
                            os.makedirs(folder_report, exist_ok=True)
                        except:
                            dprint(f'✗ directory could not be created @ {folder_name} :: Force logout user {uid}')
                            session['has_login'] = False
                            session['uid'] = uid
                            session['named'] = named
                            session['emojid'] = ''
                            return redirect(url_for('route_logout'))
                    
                        session['has_login'] = True
                        session['uid'] = uid
                        session['admind'] = admind + app.config['apac']
                        session['filed'] = os.listdir(folder_name)
                        session['reported'] = sorted(os.listdir(folder_report))
                        session['emojid'] = in_emoji 
                        session['hidden_store'] = False
                        session['hidden_storeuser'] = True
                        
                        if in_name!=named and  valid_name and  (app.config['rename']>0): 
                            session['named'] = in_name
                            db[uid][2] = in_name
                            dprint(f'⇒ {uid} ◦ {named} updated name to "{in_name}" via {request.remote_addr}') 
                            named = in_name
                        else: 
                            session['named'] = named
                            if in_name: dprint(f'⇒ {uid} ◦ {named} provided invalid name "{in_name}" (will not update)')  

                        dprint(f'● {session["uid"]} {session["emojid"]} {session["named"]} has logged in via {request.remote_addr}') 
                        return redirect(url_for('route_home'))
                    else:  
                        warn = LOGIN_FAIL_TEXT
                        msg = f'[{in_uid}] Password mismatch'                  
                else: # password not provided
                    warn = LOGIN_FAIL_TEXT
                    msg = f'[{in_uid}] Password not provided'
        else:
            warn = LOGIN_FAIL_TEXT
            msg = f'[{in_uid}] Not a valid user' 

    else:
        if session.get('has_login', False):  return redirect(url_for('route_home'))
        msg = args.welcome
        warn = LOGIN_NEED_TEXT 
        
    return render_template('login.html', msg = msg,  warn = warn)
# ------------------------------------------------------------------------------------------
# new
# ------------------------------------------------------------------------------------------
@app.route('/new', methods =['GET', 'POST'])
def route_new():
    if not app.config['reg']: return "registration is not allowed"
    if request.method == 'POST' and 'uid' in request.form and 'passwd' in request.form:
        global db
        in_uid = f"{request.form['uid']}"
        in_passwd = f"{request.form['passwd']}"
        in_name = f'{request.form["named"]}' if 'named' in request.form else ''
        in_emoji = app.config['emoji']
        in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
        valid_query, valid_name = VALIDATE_UID(in_query) , VALIDATE_NAME(in_name)
        if not valid_query:
            warn, msg = LOGIN_FAIL_TEXT, f'[{in_uid}] Not a valid user-id' 
        elif not valid_name:
            warn, msg = LOGIN_FAIL_TEXT, f'[{in_name}] Not a valid name' 
        else:
            record = db.get(in_query, None)
            if record is None: 
                if not app.config['reg']:
                    warn, msg = LOGIN_FAIL_TEXT, f'[{in_uid}] not allowed to register' 
                else:
                    admind, uid, named = app.config['reg'], in_query, in_name
                    if in_passwd: # new password provided
                        if VALIDATE_PASS(in_passwd): # new password is valid
                            db[uid] = [admind, uid, named, in_passwd]
                            warn = LOGIN_CREATE_TEXT
                            msg = f'[{in_uid}] ({named}) New password was created successfully'
                            dprint(f'● {in_uid} {in_emoji} {named} just joined via {request.remote_addr}')
            
                        else: # new password is invalid valid  
                            warn = LOGIN_NEW_TEXT
                            msg=f'[{in_uid}] New password is invalid - can use any of the alphabets (A-Z, a-z), numbers (0-9), underscore (_), dot (.) and at-symbol (@) only'
                            
                                                
                    else: #new password not provided                  
                        warn = LOGIN_NEW_TEXT
                        msg = f'[{in_uid}] New password required - can use any of the alphabets (A-Z, a-z), numbers (0-9), underscore (_), dot (.) and at-symbol (@) only'
                                            

            else:
                warn, msg = LOGIN_FAIL_TEXT, f'[{in_uid}] is already registered' 

    else:
        if session.get('has_login', False):  return redirect(url_for('route_home'))
        msg = args.register
        warn = LOGIN_REG_TEXT 
        
    return render_template('new.html', msg = msg,  warn = warn)
# ------------------------------------------------------------------------------------------
# logout
# ------------------------------------------------------------------------------------------
@app.route('/logout')
def route_logout():
    r""" logout a user and redirect to login page """
    if not session.get('has_login', False):  return redirect(url_for('route_login'))
    if not session.get('uid', False): return redirect(url_for('route_login'))
    if session['has_login']:  dprint(f'● {session["uid"]} {session["emojid"]} {session["named"]} has logged out via {request.remote_addr}') 
    else: dprint(f'✗ {session["uid"]} ◦ {session["named"]} was removed due to invalid uid ({session["uid"]}) via {request.remote_addr}') 
    session.clear()
    return redirect(url_for('route_login'))
# ------------------------------------------------------------------------------------------
# board
# ------------------------------------------------------------------------------------------
@app.route('/board', methods =['GET'])
def route_board():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'B' not in session['admind']:  return redirect(url_for('route_home'))
    if '?' in (request.args) and '+' in session['admind']: 
        if update_board(): 
            dprint(f"▶ {session['uid']} ◦ {session['named']} just refreshed the board via {request.remote_addr}")
            return redirect(url_for('route_board'))
    return BOARD_PAGE
# ------------------------------------------------------------------------------------------
# download
# ------------------------------------------------------------------------------------------
@app.route('/downloads', methods =['GET'], defaults={'req_path': ''})
@app.route('/downloads/<path:req_path>')
def route_downloads(req_path):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'D' not in session['admind']:  return redirect(url_for('route_home'))
    abs_path = os.path.join(app.config['downloads'], req_path) # Joining the base and the requested path
    if not req_path:
        if '?' in request.args and '+' in session['admind']: 
            app.config['dfl'] = GET_FILE_LIST(DOWNLOAD_FOLDER_PATH)
            dprint(f"▶ {session['uid']} ◦ {session['named']} just refreshed the download list via {request.remote_addr}")
            return redirect(url_for('route_downloads'))
    else:
        if not os.path.exists(abs_path): 
            dprint(f"⇒ requested file was not found {abs_path}") #Return 404 if path doesn't exist
            return abort(404) # (f"◦ requested file was not found") #Return 404 if path doesn't exist
        if os.path.isfile(abs_path):  #(f"◦ sending file ")
            dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the file {req_path} via {request.remote_addr}')
            return send_file(abs_path) # Check if path is a file and serve
    return render_template('downloads.html')
# ------------------------------------------------------------------------------------------
# uploads
# ------------------------------------------------------------------------------------------
@app.route('/uploads', methods =['GET'], defaults={'req_path': ''})
@app.route('/uploads/<path:req_path>')
def route_uploads(req_path):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'S' not in session['admind']:  return redirect(url_for('route_home'))
    abs_path = os.path.join(os.path.join( app.config['uploads'], session['uid']) , req_path)# Joining the base and the requested path
    if not os.path.exists(abs_path): 
        dprint(f"⇒ requested file was not found {abs_path}") #Return 404 if path doesn't exist
        return abort(404) # (f"◦ requested file was not found") #Return 404 if path doesn't exist
    if os.path.isfile(abs_path):  #(f"◦ sending file ")
        dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the file {req_path} via {request.remote_addr}')
        return send_file(abs_path) # Check if path is a file and serve
    return render_template('uploads.html')
# ------------------------------------------------------------------------------------------
# reports
# ------------------------------------------------------------------------------------------
@app.route('/reports', methods =['GET'], defaults={'req_path': ''})
@app.route('/reports/<path:req_path>')
def route_reports(req_path):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'R' not in session['admind']:  return redirect(url_for('route_home'))
    abs_path = os.path.join(os.path.join( app.config['reports'], session['uid']) , req_path)# Joining the base and the requested path
    if not os.path.exists(abs_path): 
        dprint(f"⇒ requested file was not found {abs_path}") #Return 404 if path doesn't exist
        return abort(404) # (f"◦ requested file was not found") #Return 404 if path doesn't exist
    if os.path.isfile(abs_path):  #(f"◦ sending file ")
        dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the report {req_path} via {request.remote_addr}')
        return send_file(abs_path) # Check if path is a file and serve
    return render_template('reports.html')
# ------------------------------------------------------------------------------------------
@app.route('/generate_eval_template', methods =['GET'])
def route_generate_eval_template():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if not (('X' in session['admind']) or ('+' in session['admind'])): return abort(404)
    return send_file(DICT2BUFF({k:[v[LOGIN_ORD_MAPPING["UID"]], v[LOGIN_ORD_MAPPING["NAME"]], "", "",] for k,v in db.items() if '-' not in v[LOGIN_ORD_MAPPING["ADMIN"]]} , ["UID", "NAME", "SCORE", "REMARKS"]),
                    download_name=f"eval_{app.config['topic']}_{session['uid']}.csv", as_attachment=True)
@app.route('/generate_submit_report', methods =['GET'])
def route_generate_submit_report():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if not (('X' in session['admind']) or ('+' in session['admind'])): return abort(404)
    finished_uids = set(dbsub.keys())
    remaining_uids = dbevalset.difference(finished_uids)
    absent_uids = set([puid for puid in remaining_uids if not os.path.isdir(os.path.join( app.config['uploads'], puid))])
    pending_uids = remaining_uids.difference(absent_uids)
    msg = f"Total [{len(dbevalset)}]"
    if len(dbevalset) != len(finished_uids) + len(pending_uids) + len(absent_uids): msg+=f" [!] Count Mismatch!"
    pending_uids, absent_uids, finished_uids = sorted(list(pending_uids)), sorted(list(absent_uids)), sorted(list(finished_uids))
    return \
    f"""
    <style>
    td {{padding: 10px;}}
    th {{padding: 5px;}}
    tr {{vertical-align: top;}}
    </style>
    <h3> {msg} </h3>
    <table border="1">
        <tr>
            <th>Pending [{len(pending_uids)}]</th>
            <th>Absent [{len(absent_uids)}]</th>
            <th>Finished [{len(finished_uids)}]</th>
        </tr>
        <tr>
            <td><pre>{NEWLINE.join(pending_uids)}</pre></td>
            <td><pre>{NEWLINE.join(absent_uids)}</pre></td>
            <td><pre>{NEWLINE.join(finished_uids)}</pre></td>
        </tr>
        
    </table>
    """
# ------------------------------------------------------------------------------------------
# eval
# ------------------------------------------------------------------------------------------
@app.route('/eval', methods =['GET', 'POST'])
def route_eval():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    form = UploadFileForm()
    submitter = session['uid']
    results = []
    if form.validate_on_submit():
        dprint(f"● {session['uid']} ◦ {session['named']} is trying to upload {len(form.file.data)} items via {request.remote_addr}")
        if  not ('X' in session['admind']): status, success =  "You are not allow to evaluate.", False
        else: 
            if not EVAL_XL_PATH: status, success =  "Evaluation is disabled.", False
            else:
                if len(form.file.data)!=1:  status, success = f"Expecting only one csv file", False
                else:
                    #---------------------------------------------------------------------------------
                    file = form.file.data[0]
                    isvalid, sf = VALIDATE_FILENAME_SUBMIT(secure_filename(file.filename))
                    #---------------------------------------------------------------------------------
                    if not isvalid: status, success = f"FileName is invalid '{sf}'", False
                    else:
                        try: 
                            filebuffer = BytesIO()
                            file.save(filebuffer) 
                            score_dict = BUFF2DICT(filebuffer, 0)
                            results.clear()
                            for k,v in score_dict.items():
                                in_uid = v[0] #f"{request.form['uid']}"
                                in_score = v[2] #f"{request.form['score']}"
                                in_remark = v[3]
                                if not (in_score or in_remark): continue
                                if in_score:
                                    try: _ = float(in_score)
                                    except: in_score=''
                                in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
                                valid_query = VALIDATE_UID(in_query) 
                                if not valid_query : 
                                    results.append((in_uid,f'[{in_uid}] is not a valid user.', False))
                                else: 
                                    record = db.get(in_query, None)
                                    if record is None: 
                                        results.append((in_uid,f'[{in_uid}] is not a valid user.', False))
                                    else:
                                        admind, uid, named, _ = record
                                        if ('-' in admind):
                                            results.append((in_uid,f'[{in_uid}] {named} is not in evaluation list.', False))
                                        else:
                                            scored = dbsub.get(in_query, None)                               
                                            if scored is None: # not found
                                                if not in_score:
                                                    results.append((in_uid,f'Require numeric value to assign score to [{in_uid}] {named}.', False))
                                                else:
                                                    has_req_files = GetUserFiles(uid)
                                                    if has_req_files:
                                                        dbsub[in_query] = [uid, named, in_score, in_remark, submitter]
                                                        results.append((in_uid,f'Score/Remark Created for [{in_uid}] {named}, current score is {in_score}.', True))
                                                        dprint(f"▶ {submitter} ◦ {session['named']} just evaluated {uid} ◦ {named} via {request.remote_addr}")
                                                    else:
                                                        results.append((in_uid,f'User [{in_uid}] {named} has not uploaded the required files yet.', False))
                                            else:
                                                if scored[-1] == submitter or abs(float(scored[2])) == float('inf') or ('+' in session['admind']):
                                                    if in_score:  dbsub[in_query][2] = in_score
                                                    if in_remark: dbsub[in_query][3] = in_remark
                                                    dbsub[in_query][-1] = submitter # incase of inf score
                                                    if in_score or in_remark : results.append((in_uid,f'Score/Remark Updated for [{in_uid}] {named}, current score is {dbsub[in_query][2]}. Remark is [{dbsub[in_query][3]}].', True))
                                                    else: results.append((in_uid,f'Nothing was updated for [{in_uid}] {named}, current score is {dbsub[in_query][2]}. Remark is [{dbsub[in_query][3]}].', False))
                                                    dprint(f"▶ {submitter} ◦ {session['named']} updated the evaluation for {uid} ◦ {named} via {request.remote_addr}")
                                                else:
                                                    results.append((in_uid,f'[{in_uid}] {named} has been evaluated by [{scored[-1]}], you cannot update the information. Hint: Set the score to "inf".', False))
                                                    dprint(f"▶ {submitter} ◦ {session['named']} is trying to revaluate {uid} ◦ {named} (already evaluated by [{scored[-1]}]) via {request.remote_addr}")
                            vsu = [vv for nn,kk,vv in results]
                            vsuc = vsu.count(True)
                            success = (vsuc > 0)
                            status = f'Updated {vsuc} of {len(vsu)} records'
                        except: 
                            status, success = f"Error updating scroes from file [{sf}]", False
        if success: persist_subdb()
    elif request.method == 'POST': 
        if 'uid' in request.form and 'score' in request.form:
            if EVAL_XL_PATH:
                if ('X' in session['admind']) or ('+' in session['admind']):
                    in_uid = f"{request.form['uid']}"
                    in_score = f"{request.form['score']}"
                    if in_score:
                        try: _ = float(in_score)
                        except: in_score=''
                    in_remark = f'{request.form["remark"]}' if 'remark' in request.form else ''
                    in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
                    valid_query = VALIDATE_UID(in_query) 
                    if not valid_query : 
                        status, success = f'[{in_uid}] is not a valid user.', False
                    else: 
                        record = db.get(in_query, None)
                        if record is None: 
                            status, success = f'[{in_uid}] is not a valid user.', False
                        else:
                            admind, uid, named, _ = record
                            if ('-' in admind):
                                status, success = f'[{in_uid}] {named} is not in evaluation list.', False
                            else:
                                scored = dbsub.get(in_query, None)                               
                                if scored is None: # not found
                                    if not in_score:
                                        status, success = f'Require numeric value to assign score to [{in_uid}] {named}.', False
                                    else:
                                        has_req_files = GetUserFiles(uid)
                                        if has_req_files:
                                            dbsub[in_query] = [uid, named, in_score, in_remark, submitter]
                                            status, success = f'Score/Remark Created for [{in_uid}] {named}, current score is {in_score}.', True
                                            dprint(f"▶ {submitter} ◦ {session['named']} just evaluated {uid} ◦ {named} via {request.remote_addr}")
                                        else:
                                            status, success = f'User [{in_uid}] {named} has not uploaded the required files yet.', False
                                else:
                                    if scored[-1] == submitter or abs(float(scored[2])) == float('inf') or ('+' in session['admind']):
                                        if in_score:  dbsub[in_query][2] = in_score
                                        if in_remark: dbsub[in_query][3] = in_remark
                                        dbsub[in_query][-1] = submitter # incase of inf score
                                        if in_score or in_remark : status, success =    f'Score/Remark Updated for [{in_uid}] {named}, current score is {dbsub[in_query][2]}. Remark is [{dbsub[in_query][3]}].', True
                                        else: status, success =                         f'Nothing was updated for [{in_uid}] {named}, current score is {dbsub[in_query][2]}. Remark is [{dbsub[in_query][3]}].', False
                                        dprint(f"▶ {submitter} ◦ {session['named']} updated the evaluation for {uid} ◦ {named} via {request.remote_addr}")
                                    else:
                                        status, success = f'[{in_uid}] {named} has been evaluated by [{scored[-1]}], you cannot update the information. Hint: Set the score to "inf".', False
                                        dprint(f"▶ {submitter} ◦ {session['named']} is trying to revaluate {uid} ◦ {named} (already evaluated by [{scored[-1]}]) via {request.remote_addr}")
                else: status, success =  "You are not allow to evaluate.", False
            else: status, success =  "Evaluation is disabled.", False
        else: status, success = f"You posted nothing!", False
        if success and app.config['eip']: persist_subdb()
    else:
        if ('+' in session['admind']) or ('X' in session['admind']):
            status, success = f"Eval Access is Enabled", True
        else: status, success = f"Eval Access is Disabled", False
    return render_template('evaluate.html', success=success, status=status, form=form, results=results)
# ------------------------------------------------------------------------------------------
# home - upload
# ------------------------------------------------------------------------------------------
@app.route('/home', methods =['GET', 'POST'])
def route_home():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    form = UploadFileForm()
    folder_name = os.path.join( app.config['uploads'], session['uid']) 
    if EVAL_XL_PATH:
        submitted = int(session['uid'] in dbsub)
        score = dbsub[session['uid']][2] if submitted>0 else -1
    else: submitted, score = -1, -1

    if form.validate_on_submit() and ('U' in session['admind']):
        dprint(f"● {session['uid']} ◦ {session['named']} is trying to upload {len(form.file.data)} items via {request.remote_addr}")
        if app.config['muc']==0: 
            return render_template('home.html', submitted=submitted, score=score, form=form, status=[(0, f'✗ Uploads are disabled')])
        
        if EVAL_XL_PATH:
            if submitted>0: return render_template('home.html', submitted=submitted, score=score, form=form, status=[(0, f'✗ You have been evaluated - cannot upload new files for this session.')])

        result = []
        n_success = 0
        #---------------------------------------------------------------------------------
        for file in form.file.data:
            isvalid, sf = VALIDATE_FILENAME(secure_filename(file.filename))
            isvalid = isvalid or ('+' in session['admind'])
        #---------------------------------------------------------------------------------
            
            if not isvalid:
                why_failed =  f"✗ File not accepted [{sf}] " if REQUIRED_FILES else f"✗ Extension is invalid [{sf}] "
                result.append((0, why_failed))
                continue

            file_name = os.path.join(folder_name, sf)
            if not os.path.exists(file_name):
                if len(session['filed'])>=app.config['muc']:
                    why_failed = f"✗ Upload limit reached [{sf}] "
                    result.append((0, why_failed))
                    continue
            
            try: 
                file.save(file_name) 
                why_failed = f"✓ Uploaded new file [{sf}] "
                result.append((1, why_failed))
                n_success+=1
                if sf not in session['filed']: session['filed'] = session['filed'] + [sf]
            except FileNotFoundError: 
                return redirect(url_for('route_logout'))


            

        #---------------------------------------------------------------------------------
            
        result_show = ''.join([f'\t{r[-1]}\n' for r in result])
        result_show = result_show[:-1]
        dprint(f'✓ {session["uid"]} ◦ {session["named"]} just uploaded {n_success} file(s)\n{result_show}') 
        return render_template('home.html', submitted=submitted, score=score, form=form, status=result)
    
    return render_template('home.html', submitted=submitted, score=score, form=form, status=(INITIAL_UPLOAD_STATUS if app.config['muc']!=0 else [(-1, f'Uploads are disabled')]))
# ------------------------------------------------------------------------------------------
@app.route('/uploadf', methods =['GET'])
def route_uploadf():
    r""" force upload - i.e., refresh by using os.list dir """
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    folder_name = os.path.join( app.config['uploads'], session['uid']) 
    session['filed'] = os.listdir(folder_name)
    folder_report = os.path.join(app.config['reports'], session['uid']) 
    session['reported'] = sorted(os.listdir(folder_report))
    return redirect(url_for('route_home'))
# ------------------------------------------------------------------------------------------
# purge
# ------------------------------------------------------------------------------------------
@app.route('/purge', methods =['GET'])
def route_purge():
    r""" purges all files that a user has uploaded in their respective uplaod directory
    NOTE: each user will have its won directory, so choose usernames such that a corresponding folder name is a valid one
    """
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'U' not in session['admind']:  return redirect(url_for('route_home'))
    if EVAL_XL_PATH:
        #global dbsub
        if session['uid'] in dbsub: return redirect(url_for('route_home'))

    folder_name = os.path.join( app.config['uploads'], session['uid']) 
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        for f in file_list: os.remove(os.path.join(folder_name, f))
        dprint(f'● {session["uid"]} ◦ {session["named"]} used purge via {request.remote_addr}')
        session['filed']=[]
    return redirect(url_for('route_home'))
# ------------------------------------------------------------------------------------------

class HConv: # html converter
   
    @staticmethod
    def convert(abs_path):
        new_abs_path = f'{abs_path}.html'
        if abs_path.lower().endswith(".ipynb"):
            try:
                x = __class__.nb2html( abs_path )
                with open(new_abs_path, 'w') as f: f.write(x)
                return True, (f"rendered Notebook to HTML @ {new_abs_path}")
            except: return False, (f"failed to rendered Notebook to HTML @ {new_abs_path}") 
        else: return False, (f"no renderer exists for {abs_path}")

    @staticmethod
    def remove_tag(page, tag): # does not work on nested tags
        fstart, fstop = f'<{tag}', f'/{tag}>'
        while True:
            istart = page.find(fstart)
            if istart<0: break
            istop = page[istart:].find(fstop)
            page = f'{page[:istart]}{page[istart+istop+len(fstop):]}'
        return page
    
    @staticmethod
    def nb2html(source_notebook, template_name='lab', no_script=True, html_title=None, parsed_title='Notebook',):
        #if not has_nbconvert_package: return f'<div>Requires nbconvert: python -m pip install nbconvert</div>'
        if html_title is None: # auto infer
            html_title = os.path.basename(source_notebook)
            iht = html_title.rfind('.')
            if not iht<0: html_title = html_title[:iht]
            if not html_title: html_title = (parsed_title if parsed_title else os.path.basename(os.path.dirname(source_notebook)))
        try:    
            page, _ = HTMLExporter(template_name=template_name).from_file(source_notebook,  dict(  metadata = dict( name = f'{html_title}' )    )) 
            if no_script: page = __class__.remove_tag(page, 'script') # force removing any scripts
        except: page = None
        return  page

# ------------------------------------------------------------------------------------------
# store
# ------------------------------------------------------------------------------------------
def list_store_dir(abs_path):
    dirs, files = [], []
    with os.scandir(abs_path) as it:
        for item in it:
            if item.is_file(): files.append((item.name, item.name.startswith(".")))
            elif item.is_dir(): dirs.append((item.name, item.name.startswith(".")))
            else: pass
    return dirs, files
# ------------------------------------------------------------------------------------------
@app.route('/hidden_show/<path:user_enable>', methods =['GET'])
def route_hidden_show(user_enable=''):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if len(user_enable)!=2:  return redirect(url_for('route_home'))
    if user_enable[0]=='0':
        session['hidden_store'] = (user_enable[1]!='0')
        return redirect(url_for('route_store'))
    else:
        session['hidden_storeuser'] = (user_enable[1]!='0')
        return redirect(url_for('route_storeuser'))
# ------------------------------------------------------------------------------------------
@app.route('/store', methods =['GET', 'POST'])
@app.route('/store/', methods =['GET', 'POST'])
@app.route('/store/<path:subpath>', methods =['GET', 'POST'])
def route_store(subpath=""):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if ('A' not in session['admind']) :  return abort(404)
    form = UploadFileForm()
    abs_path = os.path.join(app.config['store'], subpath)
    can_admin = (('X' in session['admind']) or ('+' in session['admind']))
    if form.validate_on_submit():
        if not can_admin: return "You cannot perform this action"
        dprint(f"● {session['uid']} ◦ {session['named']} is trying to upload {len(form.file.data)} items via {request.remote_addr}")

        result = []
        n_success = 0
        #---------------------------------------------------------------------------------
        for file in form.file.data:
            isvalid, sf = VALIDATE_FILENAME_SUBMIT(secure_filename(file.filename))
        #---------------------------------------------------------------------------------
            
            if not isvalid:
                why_failed =  f"✗ File not accepted [{sf}]"
                result.append((0, why_failed))
                continue

            file_name = os.path.join(abs_path, sf)
            #if not os.path.exists(file_name):
            #    if len(session['filed'])>=app.config['muc']:
            #        why_failed = f"✗ Upload limit reached [{sf}] "
            #        result.append((0, why_failed))
            #        continue
            
            try: 
                file.save(file_name) 
                why_failed = f"✓ Uploaded new file [{sf}] "
                result.append((1, why_failed))
                n_success+=1
                #if sf not in session['filed']: session['filed'] = session['filed'] + [sf]
            except FileNotFoundError: 
                return redirect(url_for('route_logout'))


            

        #---------------------------------------------------------------------------------
            
        result_show = ''.join([f'\t{r[-1]}\n' for r in result])
        result_show = result_show[:-1]
        dprint(f'✓ {session["uid"]} ◦ {session["named"]} just stored {n_success} file(s)\n{result_show}') 
        return redirect(url_for('route_store', subpath=subpath)) #render_template('home.html', submitted=submitted, score=score, form=form, status=result)
    else:

        if not os.path.exists(abs_path):
            if not request.args: return abort(404)
            else:
                if not can_admin: return "You cannot perform this action"
                if '?' in request.args: # create this dir

                    if "." not in os.path.basename(abs_path):
                        try:
                            os.makedirs(abs_path)
                            dprint(f"● {session['uid']} ◦ {session['named']} created new directory at {abs_path} # {subpath} via {request.remote_addr}")
                            return redirect(url_for('route_store', subpath=subpath))
                        except: return f"Error creating the directory"
                    else: return f"Directory name cannot contain (.)"
                else: return f"Invalid args for new directory"
            

        if os.path.isdir(abs_path):
            if not request.args: 
                dirs, files = list_store_dir(abs_path)
                return render_template('store.html', dirs=dirs, files=files, subpath=subpath, form=form)
            else:
                if not can_admin: return "You cannot perform this action"
                if "." not in os.path.basename(abs_path) and os.path.abspath(abs_path)!=os.path.abspath(app.config['store']): #delete this dir
                    if '!' in request.args:
                        try:
                            os.removedirs(abs_path)
                            dprint(f"● {session['uid']} ◦ {session['named']} deleted directory at {abs_path} # {subpath} via {request.remote_addr}")
                            return redirect(url_for('route_store', subpath=os.path.dirname(subpath)))
                        except:
                            return f"Error deleting the directory"
                    elif 'x' in request.args:
                        try:
                            import shutil
                            shutil.rmtree(abs_path)
                            dprint(f"● {session['uid']} ◦ {session['named']} purged directory at {abs_path} # {subpath} via {request.remote_addr}") 
                            return redirect(url_for('route_store', subpath=os.path.dirname(subpath)))
                        except:
                            return f"Error deleting the directory"

                    else: return f"Invalid args for store actions"
                else: return f"Cannot Delete this directory"
                            
        elif os.path.isfile(abs_path):
            if not request.args: 
                dprint(f"● {session['uid']} ◦ {session['named']}  viewed {abs_path} via {request.remote_addr}")
                return send_file(abs_path, as_attachment=False)
            else:
                if 'get' in request.args:
                    dprint(f"● {session['uid']} ◦ {session['named']} downloaded file at {abs_path} # {subpath} via {request.remote_addr}")
                             
                    return send_file(abs_path, as_attachment=True)
                
                elif 'del' in request.args: #delete this file
                    if not can_admin: return "You cannot perform this action"
                    try:
                        os.remove(abs_path)
                        dprint(f"● {session['uid']} ◦ {session['named']} deleted file at {abs_path} # {subpath} via {request.remote_addr}") 
                        return redirect(url_for('route_store', subpath=os.path.dirname(subpath)))
                    except:return f"Error deleting the file"
                    #else: return f"Directory name cannot contain (.)"
                else: return f"Invalid args for store actions"
                            
        
        else: return abort(404)
# ------------------------------------------------------------------------------------------
@app.route('/storeuser', methods =['GET'])
@app.route('/storeuser/', methods =['GET'])
@app.route('/storeuser/<path:subpath>', methods =['GET'])
def route_storeuser(subpath=""):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if ('X' not in session['admind']):  return abort(404)
    abs_path = os.path.join(app.config['storeuser'], subpath)
    if not os.path.exists(abs_path): return abort(404)
        
    if os.path.isdir(abs_path):
        dirs, files = list_store_dir(abs_path)
        return render_template('storeuser.html', dirs=dirs, files=files, subpath=subpath, )
    elif os.path.isfile(abs_path): 
        
        if ("html" in request.args): 
            dprint(f"● {session['uid']} ◦ {session['named']} converting to html from {subpath} via {request.remote_addr}")
            if has_nbconvert_package: hstatus, hmsg = HConv.convert(abs_path)
            else: hstatus, hmsg = False, f"missing package - nbconvert"
            
            dprint(f"{TABLINE}{'... ✓' if hstatus else '... ✗'} {hmsg}")
            return redirect(url_for('route_storeuser', subpath=os.path.dirname(subpath))) 
        else: 
            dprint(f"● {session['uid']} ◦ {session['named']} downloaded {subpath} from user-store via {request.remote_addr}")
            return send_file(abs_path, as_attachment=("get" in request.args))
    else: return abort(404)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# administrative and password reset
# ------------------------------------------------------------------------------------------

def persist_db():
    r""" writes both dbs to disk """
    global db, dbsub
    if write_logindb_to_disk(db) and write_evaldb_to_disk(dbsub): #if write_db_to_disk(db, dbsub):
        dprint(f"▶ {session['uid']} ◦ {session['named']} just persisted the db to disk via {request.remote_addr}")
        STATUS, SUCCESS = "Persisted db to disk", True
    else: STATUS, SUCCESS =  f"Write error, file might be open", False
    return STATUS, SUCCESS 

def persist_subdb():
    r""" writes eval-db to disk """
    global dbsub
    if write_evaldb_to_disk(dbsub, verbose=False): STATUS, SUCCESS = "Persisted db to disk", True
    else: STATUS, SUCCESS =  f"Write error, file might be open", False
    return STATUS, SUCCESS 

def reload_db():
    r""" reloads db from disk """
    global db, dbsub
    db = read_logindb_from_disk()
    dbsub = read_evaldb_from_disk()
    dprint(f"▶ {session['uid']} ◦ {session['named']} just reloaded the db from disk via {request.remote_addr}")
    return "Reloaded db from disk", True #  STATUS, SUCCESS

@app.route('/x/', methods =['GET'], defaults={'req_uid': ''})
@app.route('/x/<req_uid>')
def route_repassx(req_uid):
    r""" reset user password"""
    if not session.get('has_login', False): return redirect(url_for('route_login')) # "Not Allowed - Requires Login"
    form = UploadFileForm()
    results = []
    if not req_uid:
        if '+' in session['admind']: 
            if len(request.args)==1:
                if '?' in request.args: STATUS, SUCCESS = reload_db()
                elif '!' in request.args: STATUS, SUCCESS = persist_db()
                else: STATUS, SUCCESS =  f'Invalid command ({next(iter(request.args.keys()))}) ... Hint: use (?) (!) ', False
            else: 
                if len(request.args)>1: STATUS, SUCCESS =  f"Only one command is accepted ... Hint: use (?) (!) ", False
                else: STATUS, SUCCESS =  f"Admin Access is Enabled", True
        else:  STATUS, SUCCESS =  f"Admin Access is Disabled", False
    else:
        iseval, isadmin = ('X' in session['admind']), ('+' in session['admind'])
        global db
        if request.args:  
            if isadmin:
                try: 
                    in_uid = f'{req_uid}'
                    if in_uid: 
                        in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
                        valid_query = VALIDATE_UID(in_query)
                        if not valid_query: STATUS, SUCCESS = f'[{in_uid}] Not a valid user-id' , False
                        else:
                            named = request.args.get('name', "")
                            admind = request.args.get('access', "")
                            record = db.get(in_query, None)
                            if record is None: 
                                if named and admind:
                                    valid_name = VALIDATE_NAME(named)
                                    if not valid_name: STATUS, SUCCESS = f'[{named}] Requires a valid name' , False
                                    else:
                                        db[in_query] = [admind, in_query, named, '']
                                        dprint(f"▶ {session['uid']} ◦ {session['named']} just added a new user {in_query} ◦ {named} via {request.remote_addr}")
                                        STATUS, SUCCESS =  f"New User Created {in_query} {named}", True
                                else: STATUS, SUCCESS = f'Missing Arguments to create new user "{in_query}": use (name) (access)' , False
                            else:
                                STATUS, SUCCESS =  f"Updated Nothing for {in_query}", False
                                radmind, _, rnamed, _ = record
                                if admind and admind!=radmind: # trying to update access
                                    db[in_query][0] = admind
                                    dprint(f"▶ {session['uid']} ◦ {session['named']} just updated access for {in_query} from {radmind} to {admind} via {request.remote_addr}")
                                    STATUS, SUCCESS =  f"Updated Access for {in_query} from [{radmind}] to [{admind}]", True

                                if named and named!=rnamed: # trying to rename
                                    valid_name = VALIDATE_NAME(named)
                                    if not valid_name: 
                                        STATUS, SUCCESS = f'[{named}] Requires a valid name' , False
                                    else:
                                        db[in_query][2] = named
                                        dprint(f"▶ {session['uid']} ◦ {session['named']} just updated name for {in_query} from {rnamed} to {named} via {request.remote_addr}")
                                        STATUS, SUCCESS =  f"Updated Name for {in_query} from [{rnamed}] to [{named}]", True
                                
                                
                                #STATUS, SUCCESS =  f"User '{in_query}' already exists", False


                    else: STATUS, SUCCESS =  f"User-id was not provided", False
                except: STATUS, SUCCESS = f'Invalid request args ... Hint: use (name, access)'
            else: STATUS, SUCCESS =  f"Admin Access is Disabled", False
        else:
            if app.config['repass']:
                
                if iseval or isadmin:
                    in_uid = f'{req_uid}'
                    if in_uid: 
                        in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
                        record = db.get(in_query, None)
                        if record is not None: 
                            admind, uid, named, _ = record
                            if (('X' not in admind) and ('+' not in admind)) or isadmin or (session['uid']==uid):
                                db[uid][3]='' ## 3 for PASS
                                dprint(f"▶ {session['uid']} ◦ {session['named']} just reset the password for {uid} ◦ {named} via {request.remote_addr}")
                                STATUS, SUCCESS =  f"Password was reset for {uid} {named}", True
                            else: STATUS, SUCCESS =  f"You cannot reset password for account '{in_query}'", False
                        else: STATUS, SUCCESS =  f"User '{in_query}' not found", False
                    else: STATUS, SUCCESS =  f"User-id was not provided", False
                else: STATUS, SUCCESS =  "You are not allow to reset passwords", False
            else: STATUS, SUCCESS =  "Password reset is disabled for this session", False
        
    return render_template('evaluate.html',  status=STATUS, success=SUCCESS, form=form, results=results)
# ------------------------------------------------------------------------------------------

#%% [READY TO SERVE]

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DO NOT WRITE ANY NEW CODE AFTER THIS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#%% [SERVER] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def endpoints(athost):
    if athost=='0.0.0.0':
        ips=set()
        try:
            import socket
            for info in socket.getaddrinfo(socket.gethostname(), None):
                if (info[0].name == socket.AddressFamily.AF_INET.name): ips.add(info[4][0])
        except: pass
        ips=list(ips)
        ips.extend(['127.0.0.1', 'localhost'])
        return ips
    else: return [f'{athost}']

start_time = datetime.datetime.now()
sprint('◉ start server @ [{}]'.format(start_time))
for endpoint in endpoints(args.host): sprint(f'◉ http://{endpoint}:{args.port}')
serve(app, # https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html
    host = args.host,          
    port = args.port,          
    url_scheme = 'http',     
    threads = args.threads,    
    connection_limit = args.maxconnect,
    max_request_body_size = MAX_UPLOAD_SIZE,
)
end_time = datetime.datetime.now()
sprint('◉ stop server @ [{}]'.format(end_time))
sprint('↷ persisted login-db [{}]'.format(write_logindb_to_disk(db)))
sprint('↷ persisted eval-db [{}]'.format(write_evaldb_to_disk(dbsub)))

if bool(parsed.coe):
    sprint(f'↪ Cleaning up html/css templates...')
    try:
        for k in HTML_TEMPLATES_KEYS:#.items():
            h = os.path.join(HTMLDIR, f"{k}.html")
            if  os.path.isfile(h) : os.remove(h)
        #sprint(f'↪ Removing css templates @ {STATIC_DIR}')
        for k in CSS_TEMPLATES_KEYS:#.items():
            h = os.path.join(HTMLDIR, f"{k}.css")
            if os.path.isfile(h): os.remove(h)
        #os.removedirs(TEMPLATES_DIR)
        #os.removedirs(STATIC_DIR)
        sprint(f'↪ Removed html/css templates @ {HTMLDIR}')
    except:
        sprint(f'↪ Could not remove html/css templates @ {HTMLDIR}')
sprint('◉ server up-time was [{}]'.format(end_time - start_time))
sprint(f'...Finished!')
#%% [END]
# ✓
# ✗
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# author: Nelson.S
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

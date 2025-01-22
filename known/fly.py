__doc__=f""" 
-----------------------------------------------------------------------------------------------
Fly - Flask-based web app for sharing files and quiz evaluation in a single script
-----------------------------------------------------------------------------------------------
REQUIREMENTS: pip install Flask Flask-WTF waitress nbconvert beautifulsoup4

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
parser.add_argument('--dir', type=str, default='', help="path of workspace directory [DEFAULT]: current diretory")
parser.add_argument('--verbose', type=int, default=2, help="verbose level in logging (0,1,2) [DEFAULT]: 2")
parser.add_argument('--log', type=str, default='', help="name of logfile as date-time-formated string, blank by default [Note: keep blank to disable logging]") #e.g. fly_%Y_%m_%d_%H_%M_%S_%f_log.txt
parser.add_argument('--con', type=str, default='', help="config name (refers to a dict in __configs__.py - if not provided, uses 'default'")
parser.add_argument('--reg', type=str, default='', help="if specified, allow users to register with that access string such as DABU or DABUS+")
parser.add_argument('--cos', type=int, default=1, help="use 1 to create-on-start - create (overwrites) pages [DEFAULT]: 1")
parser.add_argument('--coe', type=int, default=0, help="use 1 to clean-on-exit - deletes pages [DEFAULT]: 0")
parser.add_argument('--access', type=str, default='', help="if specified, adds extra premissions to access string for this session only")
parser.add_argument('--msl', type=int, default=100, help="Max String Length for UID/NAME/PASSWORDS [DEFAULT]: 100")
parser.add_argument('--eip', type=int, default=1, help="Evaluate Immediate Persis. If True (by-default), persist the eval-db after each single evaluation (eval-db in always persisted after update from template)")
parser.add_argument('--nos', type=int, default=0, help="NoScript, use 1 to disable javascript in board files [DEFAULT]: 0")
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
    from flask import Flask, render_template, render_template_string, request, redirect, url_for, session, abort, send_file
    from flask_wtf import FlaskForm
    from wtforms import SubmitField, MultipleFileField
    from werkzeug.utils import secure_filename
    from wtforms.validators import InputRequired
    from waitress import serve
    from nbconvert import HTMLExporter 
    from bs4 import BeautifulSoup
except: exit(f'[!] The required Flask packages missing:\n  ⇒ pip install Flask Flask-WTF waitress nbconvert beautifulsoup4')

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
    for _ in range(datetime.datetime.now().microsecond % (datetime.datetime.now().second + 1)): _ = randx()
    r5 = randx()
    return ':{}:{}:{}:{}:{}:{}:'.format(r1,r2,r3,r4,r5,postfix)

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

    # --------------------------------------# general info
    topic        = "Fly",                   # topic text (main banner text)
    welcome      = "Login to Continue",     # msg shown on login page
    register     = "Register User",         # msg shown on register (new-user) page
    emoji        = "🦋",                    # emoji shown of login page and seperates uid - name
    rename       = 0,                       # if rename=1, allows users to update their names when logging in
    repass       = 1,                       # if repass=1, allows admins and evaluators to reset passwords for users - should be enabled in only one session
    reeval       = 1,                       # if reeval=1, allows evaluators to reset evaluation
    case         = 0,                       # case-sentivity level in uid
                                            #   (if case=0 uids are not converted           when matching in database)
                                            #   (if case>0 uids are converted to upper-case when matching in database)
                                            #   (if case<0 uids are converted to lower-case when matching in database)
    
    # -------------------------------------# validation
    required     = "",                     # csv list of file-names that are required to be uploaded e.g., required = "a.pdf,b.png,c.exe" (keep blank to allow all file-names)
    extra        = 1,                      # if true, allows uploading extra file (other than required)
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
    secret       = "secret.txt",      # file containing flask app secret (keep blank to generate random secret every time)
    login        = "login.csv",       # login database having four cols ADMIN, UID, NAME, PASS
    eval         = "eval.csv",        # evaluation database - created if not existing - reloads if exists
    uploads      = "uploads",         # uploads folder (uploaded files by users go here)
    reports      = "reports",         # reports folder (read-only files that are private to a user go here)
    downloads    = "downloads",       # downloads folder (public read-only access)
    store        = "store",           # store folder (public read-only, evaluators can upload and delete files)
    board        = "board.ipynb",     # board file (public read-only, a notebook displayed as a web-page)
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
                        fld_bgcolor  = "#ebebeb",
                        fld_fgcolor  = "#232323",
                        msgcolor     = "#060472",
                        
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
                        icon_delfile=   '❌',
                        icon_gethtml=   '🌐',
                        icon_hidden=    '👁️',
                        icon_gotop=     '🔝',
                                                              
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
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# RUNTIME WORKFLOW
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------



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




#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# HTML/CSS TEMPLATES
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------





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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
            <br>
            <div class="bridge">
            <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{style.logout_}'+"""</a>
            <a href="{{ url_for('route_home') }}" class="btn_home">Home</a>
            <a href="{{ url_for('route_eval') }}" class="btn_refresh">Refresh</a>
            <a href="{{ url_for('route_storeuser') }}" class="btn_store">User-Store</a>
            <a href="{{ url_for('route_generate_submit_report') }}" target="_blank" class="btn_board">User-Report</a>
            <button class="btn_reeval_large" onclick="confirm_reeval()">"""+'Reset Evaluation' + """</button>
                <script>
                    function confirm_reeval() {
                    let res = prompt("Enter UID to reset evaluation", ""); 
                    if (res != null) {
                        location.href = "{{ url_for('route_eval',req_uid='::::') }}".replace("::::", res);
                        }
                    }
                </script>
            <button class="btn_purge_large" onclick="confirm_repass()">"""+'Reset Password' + """</button>
                <script>
                    function confirm_repass() {
                    let res = prompt("Enter UID to reset password", ""); 
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
        <a href="https://github.com/auto-notify-ps/known" target="_blank"><span style="font-size: xx-large;">{{ config.emoji }}</span></a>
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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
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
                <li>
                <a href="{{ (request.path + '/' if request.path != '/' else '') + file }}"" >{{ file }}</a>
                <a href="{{ (request.path + '/' if request.path != '/' else '') + file }}?html"" target="_blank">"""+f'{style.icon_gethtml}'+"""</a>
                </li>
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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
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
                {% for (i, file, hfile) in files %}
                {% if (session.hidden_storeuser) or (not hfile) %}
                    <li>
                    <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + file, get='') }}" target="_blank">"""+f'{style.icon_getfile}'+"""</a> 
                    <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + file) }}" target="_blank">{{ file }}</a>
                    {% if file.lower().endswith('.ipynb') %}
                    <a href="{{ url_for('route_storeuser', subpath=subpath + '/' + file, html='') }}" target="_blank">"""+f'{style.icon_gethtml}'+"""</a> 
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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
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
                {% for i, file, hfile in files %}
                {% if (session.hidden_store) or (not hfile) %}
                    <li>
                    <button class="btn_del" onclick="confirm_del_{{ i }}()">"""+f'{style.icon_delfile}'+"""</button>
                    <script>
                        function confirm_del_{{ i }}() {
                        let res = confirm("Delete File?\\n\\n\\t {{ file }}");
                        if (res == true) {
                            location.href = "{{ url_for('route_store', subpath=subpath + '/' + file, del='') }}";
                            }
                        }
                    </script>
                    <span> . . . </span>
                    <a href="{{ url_for('route_store', subpath=subpath + '/' + file, get='') }}">"""+f'{style.icon_getfile}'+"""</a> 
                    <a href="{{ url_for('route_store', subpath=subpath + '/' + file) }}" target="_blank" >{{ file }}</a>
                    {% if file.lower().endswith('.ipynb') %}
                    <a href="{{ url_for('route_store', subpath=subpath + '/' + file, html='') }}" target="_blank">"""+f'{style.icon_gethtml}'+"""</a> 
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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
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
                <li>
                <a href="{{ (request.path + '/' if request.path != '/' else '') + file }}">{{ file }}</a>
                <a href="{{ (request.path + '/' if request.path != '/' else '') + file }}?html"" target="_blank">"""+f'{style.icon_gethtml}'+"""</a>
                </li>
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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
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
            <div class="userword">{{session.uid}} {{ config.emoji }} {{session.named}}</div>
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
                    {% if config.muc!=0 and not config.disableupload %}
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
                    
                {% if submitted<1 and config.muc!=0 and not config.disableupload %}
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
        background-color: {style.fld_bgcolor}; 
        color: {style.fld_fgcolor};
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

    
    .btn_del {
        padding: 2px 2px 2px;
         background-color: transparent;
        border-style: none;
        color: #FFFFFF;
        font-size: small;
        border-radius: 2px;
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

    
    .btn_reeval_large {
        padding: 2px 10px 2px;
        background-color: #8225c2; 
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
    0,0,1,0,1,0,64,64,0,0,1,0,32,0,40,66,0,0,22,0,0,0,40,0,0,0,64,0,0,0,128,0,0,0,1,0,32,0,0,0,0,0,0,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,85,85,255,3,87,87,231,73,87,87,230,150,89,89,227,170,90,90,222,161,95,92,216,105,98,98,216,13,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,143,128,120,32,146,133,118,119,149,134,112,168,151,136,105,172,154,138,101,131,157,142,
    93,52,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,87,87,228,38,87,87,230,200,87,87,230,255,88,88,228,255,91,90,223,255,93,92,217,255,96,94,212,255,99,95,206,235,101,98,200,83,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,128,128,128,2,142,130,123,126,144,131,119,249,147,133,
    113,255,150,136,108,255,153,138,102,255,156,140,97,255,159,142,91,254,161,144,87,168,162,139,70,11,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,91,91,228,28,86,86,230,227,87,87,230,255,87,87,230,255,90,89,225,255,93,91,219,255,95,94,
    214,255,98,96,208,255,101,98,203,255,104,100,197,252,108,103,193,87,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,140,128,126,140,144,131,120,255,146,133,115,255,149,135,109,255,152,137,104,255,155,139,98,255,158,141,93,255,161,144,87,255,163,146,82,255,166,148,76,195,191,128,64,4,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,87,87,230,175,87,87,230,255,87,87,
    230,255,89,89,226,255,92,91,221,255,95,93,215,255,98,95,210,255,100,97,204,255,103,99,199,255,106,102,193,255,109,104,188,236,115,102,179,20,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,140,128,128,60,143,130,122,255,146,132,116,255,148,134,111,255,151,136,105,255,154,139,100,255,157,141,94,255,160,143,89,255,163,145,83,255,165,147,78,255,168,150,73,255,170,153,
    68,117,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,87,87,233,35,87,87,230,254,87,87,230,255,88,88,228,255,91,90,222,255,94,92,217,255,97,94,211,255,100,97,206,255,102,99,200,255,105,101,195,255,108,103,189,255,111,105,184,255,114,109,179,110,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,142,129,123,174,145,131,118,255,148,134,113,255,150,136,107,255,153,138,102,255,156,140,96,255,159,142,91,255,162,145,
    85,255,165,147,80,255,167,149,74,255,170,151,69,255,173,153,63,224,255,255,0,1,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,86,86,230,92,87,87,230,255,87,87,229,255,90,89,224,255,93,92,218,255,96,94,213,255,99,96,207,255,102,98,202,255,104,100,196,255,107,103,191,255,110,105,
    185,255,113,107,180,255,116,109,175,184,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,144,131,120,231,147,133,114,255,150,135,109,255,152,137,
    103,255,155,140,98,255,158,142,92,255,161,144,87,255,164,146,81,255,167,148,76,255,169,150,70,255,172,153,65,255,175,155,59,255,180,156,52,44,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,87,87,229,109,87,87,230,255,89,89,225,255,92,91,220,255,95,93,214,255,98,95,209,255,101,98,
    203,255,104,100,198,255,106,102,192,255,109,104,187,255,112,106,182,255,115,108,176,255,118,111,171,234,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,146,132,116,248,149,135,110,255,152,137,105,255,154,139,99,255,157,141,94,255,160,143,88,255,163,145,83,255,166,148,77,255,169,150,72,255,171,152,66,255,174,154,61,255,177,156,55,255,181,159,49,93,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,87,87,229,88,89,88,227,255,91,90,
    221,255,94,93,216,255,97,95,211,255,100,97,205,255,103,99,200,255,106,101,194,255,108,103,189,255,111,106,183,255,114,108,178,255,117,110,172,255,120,112,167,255,123,113,161,27,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,148,134,112,228,151,136,106,255,154,138,101,255,156,140,95,255,159,143,90,255,162,145,84,255,165,147,79,255,168,149,73,255,171,151,68,255,173,154,62,255,176,156,
    57,255,179,158,51,255,183,159,45,141,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,87,87,228,47,91,90,223,255,93,92,218,255,96,94,212,255,99,96,207,255,102,98,201,255,105,101,196,255,108,103,190,255,110,105,185,255,113,107,179,255,116,109,174,255,119,112,168,255,122,114,163,255,124,117,
    159,74,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,149,136,108,186,153,138,102,255,156,140,97,255,158,142,91,255,161,144,86,255,164,146,80,255,167,149,
    75,255,170,151,70,255,173,153,64,255,175,155,59,255,178,157,53,255,181,159,48,255,184,161,42,188,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,102,102,204,5,93,92,219,248,95,93,214,255,98,96,208,255,101,98,203,255,104,100,197,255,107,102,192,255,110,104,186,255,112,107,181,255,115,109,
    175,255,118,111,170,255,121,113,164,255,124,115,159,255,128,117,153,122,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,152,137,103,138,155,139,99,255,158,141,
    93,255,160,144,88,255,163,146,82,255,166,148,77,255,169,150,71,255,172,152,66,255,175,154,60,255,177,157,55,255,180,159,49,255,183,161,44,255,186,163,38,236,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,95,93,215,206,97,95,210,255,100,97,204,255,103,99,199,255,106,102,
    193,255,109,104,188,255,112,106,182,255,114,108,177,255,117,110,171,255,120,112,166,255,123,115,160,255,126,117,155,255,128,119,149,169,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,154,140,101,91,157,141,95,255,160,143,89,255,163,145,84,255,165,147,78,255,168,149,73,255,171,152,67,255,174,154,62,255,177,156,56,255,179,158,51,255,182,160,45,255,185,163,40,255,188,165,34,255,193,167,
    26,29,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,97,94,
    211,158,99,97,206,255,102,99,200,255,105,101,195,255,108,103,189,255,111,105,184,255,114,107,178,255,116,110,173,255,119,112,168,255,122,114,162,255,125,116,157,255,128,118,151,255,130,121,146,217,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,154,142,95,43,159,142,91,255,162,145,85,255,165,147,80,255,167,149,74,255,170,151,69,255,173,153,63,255,176,155,58,255,179,158,52,255,181,160,
    47,255,184,162,41,255,187,164,36,255,190,166,30,255,192,169,26,77,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,99,96,209,111,101,98,202,255,104,100,197,255,107,102,191,255,110,105,186,255,113,107,180,255,116,109,175,255,118,111,169,255,121,113,164,255,124,116,158,255,127,118,153,255,130,120,
    147,255,133,122,142,253,128,128,128,12,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,170,170,85,3,161,144,87,245,164,146,81,255,167,148,76,255,169,150,70,255,172,153,
    65,255,175,155,59,255,178,157,54,255,181,159,48,255,184,161,43,255,186,163,37,255,189,166,32,255,192,168,27,255,196,169,20,125,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,101,97,202,63,103,100,198,255,106,102,193,255,109,104,187,255,112,106,182,255,115,108,176,255,118,111,171,255,120,113,
    165,255,123,115,160,255,126,117,154,255,129,119,149,255,132,121,143,255,135,124,138,255,136,128,132,58,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,163,145,
    83,202,166,148,77,255,169,150,72,255,171,152,66,255,174,154,61,255,177,156,56,255,180,158,50,255,183,161,45,255,186,163,39,255,188,165,34,255,191,167,28,255,194,169,23,255,197,172,16,172,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,96,96,207,16,105,101,194,254,108,103,189,255,111,106,
    183,255,114,108,178,255,117,110,172,255,120,112,167,255,122,114,161,255,125,116,156,255,128,119,150,255,131,121,145,255,134,123,139,255,137,125,134,255,140,128,128,106,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,166,147,79,154,168,149,74,255,171,151,68,255,173,153,63,255,176,156,57,255,179,158,52,255,182,160,46,255,185,162,41,255,188,164,35,255,190,167,30,255,193,169,24,255,196,171,
    19,255,198,173,13,221,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,77,77,230,10,88,88,221,52,92,92,217,80,96,93,213,85,99,96,207,85,102,99,
    201,85,105,102,195,85,107,103,190,238,110,105,185,255,113,107,179,255,116,109,174,255,119,111,168,255,122,114,163,255,124,116,157,255,127,118,152,255,130,120,146,255,133,122,141,255,136,125,135,255,139,127,130,255,142,129,125,182,144,132,
    120,85,147,132,114,85,150,135,108,85,153,138,102,85,156,141,96,85,159,141,93,85,162,144,87,85,165,147,81,85,166,149,74,161,170,151,70,255,173,153,64,255,175,155,59,255,178,157,53,255,181,159,48,255,184,162,42,255,187,164,
    37,255,190,166,31,255,192,168,26,255,195,170,20,255,198,172,15,255,200,174,11,254,198,176,11,90,195,177,12,85,195,177,12,85,192,180,15,85,192,180,15,85,189,180,15,85,187,183,16,64,186,186,20,26,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,77,77,230,10,87,87,230,140,90,89,225,239,92,91,
    219,255,95,93,214,255,98,96,208,255,101,98,203,255,104,100,197,255,107,102,192,255,109,104,186,255,112,106,181,255,115,109,175,255,118,111,170,255,121,113,165,255,124,115,159,255,126,117,154,255,129,120,148,255,132,122,143,255,135,124,
    137,255,138,126,132,255,141,128,126,255,143,130,121,255,146,133,115,255,149,135,110,255,152,137,104,255,155,139,99,255,158,141,93,255,160,144,88,255,163,146,82,255,166,148,77,255,169,150,71,255,172,152,66,255,175,154,60,255,177,157,
    55,255,180,159,49,255,183,161,44,255,186,163,38,255,189,165,33,255,192,167,27,255,194,170,22,255,197,172,16,255,200,174,11,255,198,175,12,255,197,176,12,255,195,178,13,255,194,179,14,255,192,180,14,255,190,181,15,255,189,182,
    16,255,187,183,16,255,186,185,17,254,184,186,18,188,184,189,20,50,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,81,81,
    228,19,87,87,230,211,89,88,226,255,92,91,221,255,95,93,215,255,97,95,210,255,100,97,204,255,103,99,199,255,106,101,194,255,109,104,188,255,111,106,183,255,114,108,177,255,117,110,172,255,120,112,166,255,123,115,161,255,126,117,
    155,255,128,119,150,255,131,121,144,255,134,123,139,255,137,125,133,255,140,128,128,255,143,130,122,255,145,132,117,255,148,134,111,255,151,136,106,255,154,139,100,255,157,141,95,255,160,143,89,255,162,145,84,255,165,147,78,255,168,149,
    73,255,171,152,67,255,174,154,62,255,177,156,56,255,179,158,51,255,182,160,45,255,185,163,40,255,188,165,34,255,191,167,29,255,194,169,23,255,196,171,18,255,199,173,13,255,199,175,11,255,197,176,12,255,196,177,13,255,194,178,
    13,255,192,180,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,184,185,17,255,183,187,18,255,181,188,19,247,181,190,18,83,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,86,86,230,183,88,88,228,255,91,90,223,255,94,92,217,255,97,94,212,255,99,96,206,255,102,99,201,255,105,101,195,255,108,103,190,255,111,105,184,255,113,107,179,255,116,110,
    173,255,119,112,168,255,122,114,162,255,125,116,157,255,128,118,151,255,130,120,146,255,133,123,140,255,136,125,135,255,139,127,129,255,142,129,124,255,145,131,118,255,147,134,113,255,150,136,107,255,153,138,102,255,156,140,96,255,159,142,
    91,255,162,144,85,255,164,147,80,255,167,149,74,255,170,151,69,255,173,153,63,255,176,155,58,255,179,158,53,255,181,160,47,255,184,162,42,255,187,164,36,255,190,166,31,255,193,168,25,255,196,171,20,255,198,173,14,255,199,175,
    11,255,198,176,12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,180,15,255,190,182,15,255,188,183,16,255,186,184,16,255,185,185,17,255,183,186,18,255,182,188,18,255,180,189,19,255,178,190,20,244,174,193,19,41,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,88,88,229,58,87,87,230,255,90,89,224,255,93,92,219,255,96,94,213,255,99,96,208,255,101,98,202,255,104,100,197,255,107,102,
    191,255,110,105,186,255,113,107,180,255,115,109,175,255,118,111,169,255,121,113,164,255,124,115,158,255,127,118,153,255,130,120,147,255,132,122,142,255,135,124,136,255,138,126,131,255,141,129,125,255,144,131,120,255,147,133,114,255,149,135,
    109,255,152,137,103,255,155,139,98,255,158,142,92,255,161,144,87,255,164,146,82,255,166,148,76,255,169,150,71,255,172,153,65,255,175,155,60,255,178,157,54,255,181,159,49,255,183,161,43,255,186,163,38,255,189,166,32,255,192,168,
    27,255,195,170,21,255,198,172,16,255,200,174,11,255,198,175,12,255,197,177,12,255,195,178,13,255,193,179,14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,181,188,
    19,255,179,190,20,255,177,191,20,255,176,192,21,158,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,87,87,230,143,89,89,226,255,92,91,220,255,95,93,215,255,98,95,
    209,255,101,97,204,255,103,100,198,255,106,102,193,255,109,104,187,255,112,106,182,255,115,108,176,255,118,110,171,255,120,113,165,255,123,115,160,255,126,117,154,255,129,119,149,255,132,121,143,255,134,124,138,255,137,126,132,255,140,128,
    127,255,143,130,122,255,146,132,116,255,149,134,111,255,151,137,105,255,154,139,100,255,157,141,94,255,160,143,89,255,163,145,83,255,166,148,78,255,168,150,72,255,171,152,67,255,174,154,61,255,177,156,56,255,180,158,50,255,183,161,
    45,255,185,163,39,255,188,165,34,255,191,167,28,255,194,169,23,255,197,172,17,255,200,174,12,255,199,175,12,255,197,176,12,255,195,177,13,255,194,179,13,255,192,180,14,255,191,181,15,255,189,182,15,255,187,183,16,255,186,184,
    17,255,184,186,17,255,183,187,18,255,181,188,19,255,179,189,19,255,178,190,20,255,176,192,21,255,175,193,21,242,255,255,0,1,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,88,88,
    227,176,91,90,222,255,94,92,216,255,97,95,211,255,100,97,205,255,103,99,200,255,105,101,194,255,108,103,189,255,111,105,183,255,114,108,178,255,117,110,172,255,120,112,167,255,122,114,161,255,125,116,156,255,128,119,151,255,131,121,
    145,255,134,123,140,255,136,125,134,255,139,127,129,255,142,129,123,255,145,132,118,255,148,134,112,255,151,136,107,255,153,138,101,255,156,140,96,255,159,143,90,255,162,145,85,255,165,147,79,255,168,149,74,255,170,151,68,255,173,153,
    63,255,176,156,57,255,179,158,52,255,182,160,46,255,185,162,41,255,187,164,35,255,190,167,30,255,193,169,24,255,196,171,19,255,199,173,13,255,199,175,11,255,197,176,12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,181,
    15,255,189,182,15,255,188,183,16,255,186,184,17,255,185,185,17,255,183,186,18,255,181,188,18,255,180,189,19,255,178,190,20,255,177,191,20,255,175,192,21,255,173,194,22,255,174,197,23,22,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,90,90,223,165,93,92,218,255,96,94,212,255,99,96,207,255,102,98,201,255,105,101,196,255,107,103,191,255,110,105,185,255,113,107,180,255,116,109,174,255,119,111,169,255,122,114,
    163,255,124,116,158,255,127,118,152,255,130,120,147,255,133,122,141,255,136,124,136,255,139,127,130,255,141,129,125,255,144,131,119,255,147,133,114,255,150,135,108,255,153,138,103,255,155,140,97,255,158,142,92,255,161,144,86,255,164,146,
    81,255,167,148,75,255,170,151,70,255,172,153,64,255,175,155,59,255,178,157,53,255,181,159,48,255,184,162,42,255,187,164,37,255,189,166,31,255,192,168,26,255,195,170,20,255,198,172,15,255,200,174,11,255,198,176,12,255,196,177,
    12,255,195,178,13,255,193,179,14,255,192,180,14,255,190,181,15,255,188,183,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,180,188,19,255,179,190,20,255,177,191,20,255,176,192,21,255,174,193,22,255,172,194,
    22,255,170,191,21,12,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,92,90,221,119,95,93,214,255,98,96,209,255,101,98,203,255,104,100,198,255,107,102,192,255,109,104,187,255,112,106,
    181,255,115,109,176,255,118,111,170,255,121,113,165,255,124,115,159,255,126,117,154,255,129,119,148,255,132,122,143,255,135,124,137,255,138,126,132,255,141,128,126,255,143,130,121,255,146,133,115,255,149,135,110,255,152,137,104,255,155,139,
    99,255,157,141,93,255,160,143,88,255,163,146,82,255,166,148,77,255,169,150,71,255,172,152,66,255,174,154,60,255,177,157,55,255,180,159,49,255,183,161,44,255,186,163,39,255,189,165,33,255,191,167,28,255,194,170,22,255,197,172,
    17,255,200,174,11,255,198,175,12,255,197,176,12,255,195,178,13,255,194,179,14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,183,16,255,186,185,17,255,184,186,17,255,182,187,18,255,181,188,19,255,179,189,19,255,178,191,
    20,255,176,192,21,255,174,193,21,255,173,194,22,255,171,194,23,219,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,89,89,211,23,97,95,210,248,100,97,205,255,103,99,
    199,255,106,101,194,255,109,104,188,255,111,106,183,255,114,108,177,255,117,110,172,255,120,112,166,255,123,114,161,255,126,117,155,255,128,119,150,255,131,121,144,255,134,123,139,255,137,125,133,255,140,128,128,255,143,130,122,255,145,132,
    117,255,148,134,111,255,151,136,106,255,154,138,100,255,157,141,95,255,160,143,89,255,162,145,84,255,165,147,79,255,168,149,73,255,171,152,68,255,174,154,62,255,176,156,57,255,179,158,51,255,182,160,46,255,185,162,40,255,188,165,
    35,255,191,167,29,255,193,169,24,255,196,171,18,255,199,173,13,255,199,175,11,255,197,176,12,255,196,177,13,255,194,178,13,255,192,180,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,184,185,17,255,183,187,
    18,255,181,188,19,255,180,189,19,255,178,190,20,255,176,191,21,255,175,193,21,255,173,194,22,255,172,195,22,255,171,197,24,118,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,99,96,207,111,102,99,201,255,105,101,195,255,108,103,190,255,111,105,184,255,113,107,179,255,116,110,173,255,119,112,168,255,122,114,162,255,125,116,157,255,128,118,151,255,130,120,146,255,133,123,140,255,136,125,
    135,255,139,127,129,255,142,129,124,255,145,131,118,255,147,133,113,255,150,136,108,255,153,138,102,255,156,140,97,255,159,142,91,255,162,144,86,255,164,147,80,255,167,149,75,255,170,151,69,255,173,153,64,255,176,155,58,255,178,157,
    53,255,181,160,47,255,184,162,42,255,187,164,36,255,190,166,31,255,193,168,25,255,195,171,20,255,198,173,14,255,199,174,11,255,198,176,12,255,196,177,13,255,195,178,13,255,193,179,14,255,191,180,14,255,190,182,15,255,188,183,
    16,255,187,184,16,255,185,185,17,255,183,186,18,255,182,187,18,255,180,189,19,255,179,190,20,255,177,191,20,255,175,192,21,255,174,193,22,255,172,195,22,255,170,196,23,208,170,198,28,9,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,105,101,197,124,107,102,191,252,110,105,186,255,113,107,180,255,115,109,175,255,118,111,169,255,121,113,164,255,124,115,158,255,127,118,
    153,255,130,120,148,255,132,122,142,255,135,124,137,255,138,126,131,255,141,128,126,255,144,131,120,255,147,133,115,255,149,135,109,255,152,137,104,255,155,139,98,255,158,142,93,255,161,144,87,255,164,146,82,255,166,148,76,255,169,150,
    71,255,172,152,65,255,175,155,60,255,178,157,54,255,181,159,49,255,183,161,43,255,186,163,38,255,189,166,32,255,192,168,27,255,195,170,21,255,197,172,16,255,200,174,11,255,198,175,12,255,197,176,12,255,195,178,13,255,193,179,
    14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,181,188,19,255,179,189,19,255,177,191,20,255,176,192,21,255,174,193,21,255,173,194,22,255,170,195,23,196,167,196,
    20,26,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,111,105,188,46,111,106,182,151,115,109,177,209,117,110,
    171,235,120,113,166,238,123,115,160,238,126,117,155,238,129,119,149,238,132,121,144,238,134,123,138,238,137,126,133,247,140,128,127,255,143,130,122,255,146,132,116,255,149,134,111,255,151,137,105,255,154,139,100,255,157,141,94,255,160,143,
    89,255,163,145,83,255,166,147,78,255,168,150,72,255,171,152,67,254,174,154,61,238,177,156,56,238,180,158,50,238,183,161,45,238,185,163,39,238,188,165,34,238,191,167,28,238,194,169,23,238,197,171,17,239,199,174,12,255,199,175,
    12,255,197,176,12,255,195,177,13,255,194,179,13,255,192,180,14,255,191,181,15,255,189,182,15,255,187,183,16,255,186,184,17,255,184,186,17,255,183,187,18,255,181,188,19,245,179,189,19,238,178,190,20,238,176,192,21,238,175,194,
    21,220,172,193,22,182,170,196,23,90,191,191,0,4,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,139,128,130,112,142,129,123,255,145,132,118,255,148,134,112,255,151,136,
    107,255,153,138,101,255,156,140,96,255,159,142,90,255,162,145,85,255,165,147,79,255,168,149,74,255,170,151,68,255,173,153,63,253,170,149,64,12,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,191,191,0,4,199,176,11,247,198,176,12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,181,15,255,190,182,15,255,188,183,16,255,186,184,17,255,185,185,17,255,183,186,18,255,182,188,18,255,180,190,
    18,125,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,139,128,
    124,64,144,131,119,255,147,133,114,255,150,135,108,255,153,137,103,255,155,140,97,255,158,142,92,255,161,144,86,255,164,146,81,255,167,148,75,255,170,151,70,255,172,153,65,255,175,155,59,255,179,157,54,57,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,198,175,13,203,196,177,12,255,195,178,13,255,193,179,14,255,192,180,14,255,190,181,15,255,188,183,16,255,187,184,16,255,185,185,
    17,255,184,186,18,255,182,187,18,255,180,188,19,255,179,190,19,172,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,143,128,128,16,146,132,115,254,149,135,110,255,152,137,105,255,155,139,99,255,157,141,94,255,160,143,88,255,163,146,83,255,166,148,77,255,169,150,72,255,172,152,66,255,174,154,
    61,255,177,156,55,255,180,158,51,105,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,197,176,12,155,195,177,13,255,194,179,14,255,192,180,14,255,190,181,
    15,255,189,182,15,255,187,183,16,255,186,185,17,255,184,186,17,255,182,187,18,255,181,188,19,255,179,189,19,255,179,190,20,220,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,148,134,112,223,151,136,106,255,154,138,101,255,157,141,95,255,159,143,90,255,162,145,84,255,165,147,
    79,255,168,149,73,255,171,151,68,255,174,154,62,255,176,156,57,255,179,158,51,255,181,159,45,152,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,196,177,
    14,108,194,178,13,255,193,180,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,185,185,17,255,183,187,18,255,181,188,19,255,180,189,19,255,178,190,20,255,177,191,20,254,170,187,17,15,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,150,136,108,175,153,138,102,255,156,140,
    97,255,159,142,91,255,161,144,86,255,164,146,80,255,167,149,75,255,170,151,69,255,173,153,64,255,176,155,58,255,178,157,53,255,181,160,47,255,184,162,42,200,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,196,179,13,60,193,179,14,255,191,180,14,255,190,182,15,255,188,183,16,255,187,184,16,255,185,185,17,255,183,186,18,255,182,187,18,255,180,189,19,255,179,190,20,255,177,191,
    20,255,175,192,21,255,176,192,21,61,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,151,137,104,128,155,139,98,255,158,141,93,255,161,144,87,255,163,146,82,255,166,148,76,255,169,150,71,255,172,152,65,255,175,155,60,255,178,157,54,255,180,159,49,255,183,161,43,255,186,163,38,246,170,170,
    0,3,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,196,177,20,13,192,180,14,253,190,181,15,255,189,182,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,
    18,255,181,188,19,255,179,189,19,255,177,191,20,255,176,192,21,255,174,193,21,255,173,194,21,109,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,153,140,99,80,157,141,94,255,160,143,89,255,163,145,83,255,165,147,78,255,168,150,72,255,171,152,67,255,174,154,62,255,177,156,56,255,180,158,
    51,255,182,160,45,255,185,163,40,255,188,165,34,255,193,168,31,41,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,191,180,15,219,189,182,15,255,188,183,
    16,255,186,184,17,255,184,186,17,255,183,187,18,255,181,188,19,255,179,189,19,255,178,190,20,255,176,191,21,255,175,193,21,255,173,194,22,255,172,195,23,156,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,144,128,122,46,146,131,118,84,148,135,113,102,150,135,108,102,153,138,100,102,156,140,95,126,159,142,91,255,162,145,85,255,165,147,80,255,167,149,74,255,170,151,
    69,255,173,153,63,255,176,155,58,255,179,158,52,255,182,160,47,255,184,162,41,255,187,164,36,255,190,166,30,255,194,168,26,150,195,170,20,102,200,173,15,102,200,175,10,102,198,175,13,102,195,178,13,102,195,178,13,102,193,180,
    15,102,190,180,15,102,189,182,15,210,188,183,16,255,186,184,16,255,185,185,17,255,183,186,18,255,182,188,18,255,180,189,19,255,178,190,20,255,177,191,20,255,175,192,21,255,174,193,22,255,172,195,22,255,170,196,23,220,170,198,
    25,102,168,198,25,102,165,200,25,102,165,200,25,102,163,203,25,102,160,203,28,102,160,203,26,99,159,205,28,72,153,204,26,20,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,139,128,131,66,141,129,125,208,144,131,120,255,147,133,114,255,150,135,109,255,152,137,103,255,155,140,98,255,158,142,92,255,161,144,
    87,255,164,146,81,255,167,148,76,255,169,150,70,255,172,153,65,255,175,155,59,255,178,157,54,255,181,159,48,255,184,161,43,255,186,164,37,255,189,166,32,255,192,168,26,255,195,170,21,255,198,172,15,255,200,174,11,255,198,175,
    12,255,196,177,12,255,195,178,13,255,193,179,14,255,192,180,14,255,190,181,15,255,188,183,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,180,188,19,255,179,190,20,255,177,191,20,255,176,192,21,255,174,193,
    21,255,172,194,22,255,171,196,23,255,169,197,23,255,168,198,24,255,166,199,25,255,164,200,25,255,163,201,26,255,161,203,27,255,160,204,27,255,158,205,28,255,156,206,29,255,155,207,29,248,153,209,30,160,150,210,30,17,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,138,125,133,100,140,128,128,252,143,130,121,255,146,132,116,255,149,135,110,255,152,137,
    105,255,154,139,99,255,157,141,94,255,160,143,88,255,163,146,83,255,166,148,77,255,169,150,72,255,171,152,66,255,174,154,61,255,177,156,55,255,180,159,50,255,183,161,44,255,186,163,39,255,188,165,33,255,191,167,28,255,194,169,
    22,255,197,172,17,255,200,174,11,255,199,175,12,255,197,176,12,255,195,177,13,255,194,179,14,255,192,180,14,255,191,181,15,255,189,182,15,255,187,183,16,255,186,185,17,255,184,186,17,255,182,187,18,255,181,188,19,255,179,189,
    19,255,178,190,20,255,176,192,21,255,174,193,21,255,173,194,22,255,171,195,23,255,170,196,23,255,168,198,24,255,166,199,25,255,165,200,25,255,163,201,26,255,162,202,26,255,160,203,27,255,158,205,28,255,157,206,28,255,155,207,
    29,255,154,208,30,255,152,209,30,255,150,211,31,222,147,216,29,26,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,138,128,133,50,140,127,128,249,142,130,
    123,255,145,132,117,255,148,134,112,255,151,136,106,255,154,138,101,255,157,141,95,255,159,143,90,255,162,145,84,255,165,147,79,255,168,149,73,255,171,151,68,255,173,154,62,255,176,156,57,255,179,158,51,255,182,160,46,255,185,162,
    40,255,188,164,35,255,190,167,29,255,193,169,24,255,196,171,19,255,199,173,13,255,199,175,11,255,197,176,12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,185,185,
    17,255,183,187,18,255,181,188,19,255,180,189,19,255,178,190,20,255,177,191,20,255,175,192,21,255,173,194,22,255,172,195,22,255,170,196,23,255,169,197,24,255,167,198,24,255,165,200,25,255,164,201,26,255,162,202,26,255,161,203,
    27,255,159,204,28,255,157,205,28,255,156,207,29,255,154,208,30,255,153,209,30,255,151,210,31,255,149,211,31,255,148,213,31,195,255,255,0,1,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,139,127,130,169,142,129,124,255,144,131,119,255,147,133,113,255,150,136,108,255,153,138,102,255,156,140,97,255,159,142,91,255,161,144,86,255,164,146,80,255,167,149,75,255,170,151,69,255,173,153,64,255,175,155,
    58,255,178,157,53,255,181,159,48,255,184,162,42,255,187,164,37,255,190,166,31,255,192,168,26,255,195,170,20,255,198,173,15,255,199,174,11,255,198,176,12,255,196,177,13,255,195,178,13,255,193,179,14,255,191,180,14,255,190,181,
    15,255,188,183,16,255,187,184,16,255,185,185,17,255,183,186,18,255,182,187,18,255,180,189,19,255,179,190,20,255,177,191,20,255,175,192,21,255,174,193,22,255,172,194,22,255,171,196,23,255,169,197,24,255,167,198,24,255,166,199,
    25,255,164,200,25,255,163,202,26,255,161,203,27,255,159,204,27,255,158,205,28,255,156,206,29,255,155,207,29,255,153,209,30,255,151,210,31,255,150,211,31,255,148,212,32,255,147,213,33,255,143,215,32,64,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,128,128,128,2,141,128,126,247,144,131,120,255,146,133,115,255,149,135,109,255,152,137,104,255,155,139,98,255,158,141,93,255,161,144,88,255,163,146,82,255,166,148,
    77,255,169,150,71,255,172,152,66,255,175,155,60,255,178,157,55,255,180,159,49,255,183,161,44,255,186,163,38,255,189,165,33,255,192,168,27,255,194,170,22,255,197,172,16,255,200,174,11,255,198,175,12,255,197,176,12,255,195,178,
    13,255,194,179,14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,181,188,19,255,179,189,19,255,177,191,20,255,176,192,21,255,174,193,21,255,173,194,22,255,171,195,
    23,255,169,196,23,255,168,198,24,255,166,199,25,255,165,200,25,255,163,201,26,255,161,202,27,255,160,204,27,255,158,205,28,255,157,206,28,255,155,207,29,255,153,208,30,255,152,209,30,255,150,211,31,255,149,212,32,255,147,213,
    32,255,145,214,33,255,143,214,34,144,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,143,133,122,25,143,130,122,255,146,132,117,255,148,134,111,255,151,136,106,255,154,139,100,255,157,141,
    95,255,160,143,89,255,163,145,84,255,165,147,78,255,168,150,73,255,171,152,67,255,174,154,62,255,177,156,56,255,180,158,51,255,182,160,45,255,185,163,40,255,188,165,34,255,191,167,29,255,194,169,23,255,196,171,18,255,199,173,
    12,255,199,175,11,255,197,176,12,255,196,177,13,255,194,178,13,255,192,180,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,184,186,17,255,183,187,18,255,181,188,19,255,180,189,19,255,178,190,20,255,176,191,
    21,255,175,193,21,255,173,194,22,255,172,195,22,255,170,196,23,255,168,197,24,255,167,199,24,255,165,200,25,255,164,201,26,255,162,202,26,255,160,203,27,255,159,204,28,255,157,206,28,255,156,207,29,255,154,208,30,255,152,209,
    30,255,151,210,31,255,149,212,32,255,147,213,32,255,146,214,33,255,144,215,33,255,143,216,33,176,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,139,139,116,11,145,131,118,254,148,134,
    113,255,150,136,107,255,153,138,102,255,156,140,96,255,159,142,91,255,162,145,85,255,165,147,80,255,167,149,74,255,170,151,69,255,173,153,63,255,176,155,58,255,179,158,52,255,182,160,47,255,184,162,41,255,187,164,36,255,190,166,
    30,255,193,168,25,255,196,171,19,255,199,173,14,255,199,175,11,255,198,176,12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,180,15,255,190,182,15,255,188,183,16,255,186,184,16,255,185,185,17,255,183,186,18,255,182,188,
    18,255,180,189,19,255,178,190,20,255,177,191,20,255,175,192,21,255,174,193,22,255,172,195,22,255,170,196,23,255,169,197,24,255,167,198,24,255,166,199,25,255,164,201,26,255,162,202,26,255,161,203,27,255,159,204,27,255,158,205,
    28,255,156,206,29,255,154,208,29,255,153,209,30,255,151,210,31,255,150,211,31,255,148,212,32,255,146,214,33,255,145,215,33,255,143,216,34,255,143,217,34,159,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,147,133,114,215,150,135,109,255,152,137,103,255,155,140,98,255,158,142,92,255,161,144,87,255,164,146,81,255,167,148,76,255,169,150,70,255,172,153,65,255,175,155,59,255,178,157,54,255,181,159,
    48,255,184,161,43,255,186,163,37,255,189,166,32,255,192,168,26,255,195,170,21,255,198,172,15,255,200,174,11,255,198,175,12,255,197,177,12,255,195,178,13,255,193,179,14,255,192,180,14,255,190,181,15,255,188,182,16,255,187,184,
    16,255,185,185,17,255,184,186,18,255,182,187,18,255,180,188,19,255,179,190,20,255,177,191,20,255,176,192,21,255,174,193,21,255,172,194,22,255,171,195,23,255,169,197,23,255,168,198,24,255,166,199,25,255,164,200,25,255,163,201,
    26,255,161,203,27,255,160,204,27,255,158,205,28,255,156,206,29,255,155,207,29,255,153,208,30,255,152,210,31,255,150,211,31,255,148,212,32,255,147,213,32,255,145,214,33,255,144,216,34,255,143,216,34,255,144,216,35,110,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,150,136,110,109,152,137,105,255,154,139,99,255,157,141,94,255,160,143,88,255,163,145,83,255,166,148,77,255,169,150,72,255,171,152,
    66,255,174,154,61,255,177,156,55,255,180,159,50,255,183,161,45,255,186,163,39,255,188,165,34,255,191,167,28,255,194,169,23,255,197,172,17,255,200,174,12,255,199,175,12,255,197,176,12,255,195,177,13,255,194,179,14,255,192,180,
    14,255,191,181,15,255,189,182,15,255,187,183,16,255,186,185,17,255,184,186,17,255,183,187,18,255,181,188,19,255,179,189,19,255,178,190,20,255,176,192,21,255,175,193,21,255,173,194,22,255,171,195,23,255,170,196,23,255,168,197,
    24,255,167,199,25,255,165,200,25,255,163,201,26,255,162,202,26,255,160,203,27,255,159,205,28,255,157,206,28,255,155,207,29,255,154,208,30,255,152,209,30,255,150,210,31,255,149,212,32,255,147,213,32,255,146,214,33,255,144,215,
    34,255,143,216,34,255,143,216,34,244,150,210,30,17,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,128,128,128,4,155,137,100,193,156,140,95,255,159,143,90,255,162,145,
    84,255,165,147,79,255,168,149,74,255,171,151,68,255,173,154,63,255,176,156,57,255,179,158,52,255,182,160,46,255,185,162,41,255,188,164,35,255,190,167,30,255,193,169,24,255,196,171,19,255,199,173,13,255,199,175,11,255,197,176,
    12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,185,185,17,255,183,187,18,255,181,188,19,255,180,189,19,255,178,190,20,255,177,191,20,255,175,192,21,255,173,194,
    22,255,172,195,22,255,170,196,23,255,169,197,24,255,167,198,24,255,165,200,25,255,164,201,26,255,162,202,26,255,161,203,27,255,159,204,28,255,157,205,28,255,156,207,29,255,154,208,29,255,153,209,30,255,151,210,31,255,149,211,
    31,255,148,213,32,255,146,214,33,255,145,215,33,255,143,216,34,255,143,216,34,255,142,216,34,97,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,161,134,94,19,159,142,92,180,161,144,86,255,164,146,81,255,167,149,75,255,170,151,70,255,173,153,64,255,175,155,59,255,178,157,53,255,181,159,48,255,184,162,42,255,187,164,37,255,190,166,31,255,192,168,26,255,195,170,
    20,255,198,172,15,255,200,174,11,255,198,176,12,255,196,177,12,255,195,178,13,255,193,179,14,255,191,180,14,255,190,181,15,255,188,183,16,255,187,184,16,255,185,185,17,255,183,186,18,255,182,187,18,255,180,189,19,255,179,190,
    20,255,177,191,20,255,175,192,21,255,174,193,22,255,172,194,22,255,171,196,23,255,169,197,23,255,167,198,24,255,166,199,25,255,164,200,25,255,163,202,26,255,161,203,27,255,159,204,27,255,158,205,28,255,156,206,29,255,155,207,
    29,255,153,209,30,255,151,210,31,255,150,211,31,255,148,212,32,255,147,213,33,255,145,215,33,255,143,216,34,255,144,216,34,247,144,215,35,103,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,0,1,162,147,83,71,166,149,77,163,170,150,71,200,172,152,66,221,175,153,60,221,177,157,55,221,180,159,48,221,183,162,44,221,186,163,
    38,221,189,165,33,248,192,168,27,255,194,170,22,255,197,172,16,255,200,174,11,255,198,175,12,255,197,176,12,255,195,178,13,255,194,179,14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,183,16,244,186,185,17,221,183,186,
    17,221,182,187,18,221,181,188,18,221,179,189,18,221,178,192,20,221,177,192,21,221,174,193,21,221,173,195,22,232,171,195,23,255,170,196,23,255,168,198,24,255,166,199,25,255,165,200,25,255,163,201,26,255,162,202,27,255,160,204,
    27,255,158,205,28,255,157,206,28,255,155,207,29,255,153,208,30,255,152,209,30,226,150,211,31,221,149,212,32,221,147,213,32,221,145,214,33,216,144,215,34,189,142,216,33,131,140,214,33,31,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,191,167,29,174,194,169,23,255,196,171,18,255,199,173,12,255,199,175,11,255,197,176,12,255,196,177,13,255,194,178,13,255,192,180,14,255,191,181,15,255,189,182,
    15,255,188,183,16,255,186,184,16,201,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,173,194,22,59,170,196,23,255,168,197,24,255,167,198,24,255,165,200,
    25,255,164,201,26,255,162,202,26,255,160,203,27,255,159,204,28,255,157,206,28,255,156,207,29,255,154,208,30,255,152,209,30,255,150,209,29,61,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,193,169,24,127,196,171,20,255,198,173,14,255,199,175,11,255,198,176,12,255,196,177,13,255,194,178,
    13,255,193,179,14,255,191,180,15,255,190,182,15,255,188,183,16,255,186,184,16,255,185,185,17,246,191,191,0,4,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,170,191,
    21,12,169,197,24,253,167,198,24,255,166,199,25,255,164,201,26,255,162,202,26,255,161,203,27,255,159,204,27,255,158,205,28,255,156,206,29,255,154,208,29,255,153,209,30,255,151,210,31,255,150,211,30,109,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,194,171,23,79,198,172,16,255,200,174,
    11,255,198,175,12,255,197,177,12,255,195,178,13,255,193,179,14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,188,18,42,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,168,198,25,218,166,199,25,255,164,200,25,255,163,201,26,255,161,203,27,255,160,204,27,255,158,205,28,255,156,206,29,255,155,207,29,255,153,208,30,255,152,210,
    30,255,150,211,31,255,149,213,33,156,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,197,173,16,31,200,174,12,255,199,175,12,255,197,176,12,255,195,177,13,255,194,179,13,255,192,180,14,255,191,181,15,255,189,182,15,255,187,183,16,255,186,184,17,255,184,186,17,255,183,187,18,255,181,187,
    20,90,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,167,200,24,170,165,200,25,255,163,201,26,255,162,202,26,255,160,203,27,255,159,205,28,255,157,206,
    28,255,155,207,29,255,154,208,30,255,152,209,30,255,151,210,31,255,149,212,32,255,148,213,33,204,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,199,175,11,237,197,176,12,255,196,177,13,255,194,178,13,255,193,179,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,
    17,255,185,185,17,255,183,186,18,255,181,188,18,255,181,190,19,137,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,166,199,25,123,164,201,26,255,162,202,
    26,255,161,203,27,255,159,204,28,255,157,205,28,255,156,207,29,255,154,208,29,255,153,209,30,255,151,210,31,255,149,211,31,255,148,212,32,255,146,214,33,248,153,204,51,5,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,199,176,12,190,196,177,12,255,195,178,13,255,193,179,14,255,192,180,
    14,255,190,181,15,255,188,183,16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,180,189,19,255,179,190,21,185,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,164,201,23,76,163,201,26,255,161,203,27,255,159,204,27,255,158,205,28,255,156,206,29,255,155,207,29,255,153,209,30,255,151,210,31,255,150,211,31,255,148,212,32,255,147,213,33,255,145,214,33,255,142,215,
    34,45,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,197,175,
    13,141,195,178,13,255,194,179,14,255,192,180,14,255,190,181,15,255,189,182,16,255,187,183,16,255,186,185,17,255,184,186,17,255,182,187,18,255,181,188,19,255,179,189,19,255,178,191,20,228,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,161,198,28,27,162,202,27,255,160,204,27,255,158,205,28,255,157,206,28,255,155,207,29,255,154,208,30,255,152,209,30,255,150,211,31,255,149,212,
    32,255,147,213,32,255,146,214,33,255,144,215,34,255,143,215,34,89,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,195,176,14,94,194,178,13,255,192,180,14,255,191,181,15,255,189,182,15,255,188,183,16,255,186,184,17,255,184,185,17,255,183,187,18,255,181,188,19,255,180,189,19,255,178,190,
    20,255,176,191,21,250,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,160,203,27,234,159,204,28,255,157,206,28,255,156,207,29,255,154,208,
    30,255,152,209,30,255,151,210,31,255,149,211,32,255,148,213,32,255,146,214,33,255,144,215,33,255,143,216,34,255,144,216,35,110,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,193,176,11,45,193,179,14,255,191,180,14,255,190,182,15,255,188,183,16,255,187,184,16,255,185,185,17,255,183,186,
    18,255,182,187,18,255,180,189,19,255,179,190,20,255,177,191,20,255,176,192,21,231,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,159,204,
    28,185,158,205,28,255,156,206,29,255,154,208,29,255,153,209,30,255,151,210,31,255,150,211,31,255,148,212,32,255,146,213,33,255,145,215,33,255,143,216,34,255,143,216,34,255,144,216,33,92,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,128,0,2,192,179,14,233,190,181,15,255,189,182,
    16,255,187,184,16,255,185,185,17,255,184,186,18,255,182,187,18,255,181,188,19,255,179,189,19,255,177,191,20,255,176,192,21,255,173,193,21,172,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,157,204,28,120,157,206,29,255,155,207,29,255,153,208,30,255,152,210,30,255,150,211,31,255,149,212,32,255,147,213,32,255,145,214,33,255,144,215,34,255,143,216,34,255,143,216,
    34,254,143,218,38,34,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,190,180,14,126,189,182,15,255,187,183,16,255,186,184,17,255,184,186,17,255,183,187,18,255,181,188,19,255,179,189,19,255,178,190,20,255,176,192,21,255,175,193,21,254,174,194,20,63,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,155,211,33,23,155,207,29,243,154,208,30,255,152,209,30,255,151,210,31,255,149,212,32,255,147,213,32,255,146,214,
    33,255,144,215,34,255,143,216,34,255,143,216,34,255,143,216,34,178,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,182,182,0,7,188,183,16,202,186,184,17,255,185,185,17,255,183,186,18,255,182,188,18,255,180,189,19,255,178,190,20,255,177,191,20,255,175,192,
    21,255,172,193,23,145,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,153,207,30,95,153,209,30,254,151,210,
    31,255,149,211,31,255,148,212,32,255,146,214,33,255,145,215,33,255,143,216,34,255,143,216,34,255,144,216,34,231,141,220,35,29,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,180,180,15,17,184,184,17,177,184,186,18,255,182,187,18,255,180,188,
    19,255,179,190,20,255,177,191,20,255,176,192,21,250,173,193,21,131,255,255,0,1,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,152,211,30,92,150,211,31,241,148,212,32,255,147,213,33,255,145,214,33,255,144,216,34,255,143,216,34,255,143,216,34,205,145,214,34,37,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,182,186,17,59,181,189,20,142,179,188,18,180,179,191,20,170,176,193,21,123,175,189,22,35,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,148,215,27,19,147,214,32,111,147,214,34,167,144,215,34,179,143,217,33,153,144,216,33,78,128,255,0,2,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,
    255,0,255,255,255,0,255,255,255,255,255,255,255,255,255,248,15,255,224,127,255,255,255,240,7,255,128,31,255,255,255,224,3,255,128,15,255,255,255,224,1,255,0,15,255,255,255,192,1,255,0,7,255,255,255,192,1,255,0,7,
    255,255,255,192,1,255,0,7,255,255,255,192,0,255,0,7,255,255,255,192,0,255,0,7,255,255,255,192,0,255,0,7,255,255,255,224,0,255,0,3,255,255,255,224,0,255,0,3,255,255,255,224,0,127,0,3,255,255,255,224,
    0,127,128,3,255,255,255,224,0,127,128,3,255,255,248,0,0,0,0,0,3,255,224,0,0,0,0,0,0,255,192,0,0,0,0,0,0,127,192,0,0,0,0,0,0,63,128,0,0,0,0,0,0,63,128,0,0,0,0,0,
    0,31,128,0,0,0,0,0,0,31,128,0,0,0,0,0,0,31,128,0,0,0,0,0,0,63,128,0,0,0,0,0,0,63,192,0,0,0,0,0,0,63,224,0,0,0,0,0,0,127,240,0,0,0,0,0,0,255,255,252,
    0,15,224,0,127,255,255,252,0,15,240,0,127,255,255,252,0,15,240,0,127,255,255,254,0,15,240,0,63,255,255,254,0,15,240,0,63,255,255,254,0,7,240,0,63,255,255,254,0,7,248,0,63,255,255,192,0,0,0,0,
    0,31,255,0,0,0,0,0,0,7,254,0,0,0,0,0,0,3,252,0,0,0,0,0,0,1,252,0,0,0,0,0,0,1,248,0,0,0,0,0,0,1,248,0,0,0,0,0,0,1,248,0,0,0,0,0,0,1,252,0,
    0,0,0,0,0,1,252,0,0,0,0,0,0,1,252,0,0,0,0,0,0,3,254,0,0,0,0,0,0,7,255,0,0,0,0,0,0,15,255,255,192,1,254,0,7,255,255,255,192,0,254,0,7,255,255,255,192,0,255,0,
    7,255,255,255,192,0,255,0,7,255,255,255,224,0,255,0,3,255,255,255,224,0,255,0,3,255,255,255,224,0,255,0,3,255,255,255,224,0,255,128,3,255,255,255,224,0,255,128,3,255,255,255,224,0,255,128,3,255,255,255,
    240,0,255,128,7,255,255,255,240,1,255,192,7,255,255,255,248,1,255,224,15,255,255,255,254,7,255,240,31,255,255,255,255,255,255,255,255,255,
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
dbevaluatorset = sorted(list(set([k for k,v in db.items() if 'X' in v[0]])))
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
app.config['disableupload'] = False
app.config['board'] =     (BOARD_FILE_MD is not None)
app.config['reg'] =       (parsed.reg)
app.config['repass'] =    bool(args.repass)
app.config['reeval'] =    bool(args.reeval)
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
                            return redirect(url_for('route_logout'))
                    
                        session['has_login'] = True
                        session['uid'] = uid
                        session['admind'] = admind + app.config['apac']
                        session['filed'] = os.listdir(folder_name)
                        session['reported'] = sorted(os.listdir(folder_report))
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

                        dprint(f'● {session["uid"]} {app.config["emoji"]} {session["named"]} has logged in via {request.remote_addr}') 
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
    if session['has_login']:  dprint(f'● {session["uid"]} {app.config["emoji"]} {session["named"]} has logged out via {request.remote_addr}') 
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
            if ("html" in request.args): 
                dprint(f"● {session['uid']} ◦ {session['named']} converting to html from {req_path} via {request.remote_addr}")
                try:
                    hstatus, hmsg = HConv.convertx(abs_path)
                except: hstatus, hmsg = False, f"Exception while converting {req_path} to a web-page"
                
                #dprint(f"{TABLINE}{'... ✓' if hstatus else '... ✗'} {hmsg}")
                return hmsg #if hstatus else  send_file(abs_path, as_attachment=True)
            else: 
                dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the file {req_path} via {request.remote_addr}')
                return send_file(abs_path, as_attachment=False) # Check if path is a file and serve
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
        if ("html" in request.args): 
            dprint(f"● {session['uid']} ◦ {session['named']} converting to html from {req_path} via {request.remote_addr}")
            try:
                hstatus, hmsg = HConv.convertx(abs_path)
            except: hstatus, hmsg = False, f"Exception while converting {req_path} to a web-page"
            
            #dprint(f"{TABLINE}{'... ✓' if hstatus else '... ✗'} {hmsg}")
            return hmsg #if hstatus else  send_file(abs_path, as_attachment=True)
        else: 
            dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the file {req_path} via {request.remote_addr}')
            return send_file(abs_path, as_attachment=False) # Check if path is a file and serve
        
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
    #{NEWLINE.join(finished_uids)} {NEWLINE.join(pending_uids)} {NEWLINE.join(absent_uids)}
    
    
    htable0="""
    <html>
        <head>
            <meta charset="UTF-8">
            <title> Live Report {{ config.topic }} </title>
            <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
        </head>
        <body>
    """ + f"""
    <style>
    td {{padding: 10px;}}
    th {{padding: 5px;}}
    tr {{vertical-align: top;}}
    </style>
    <h2> {msg} </h2>
    <table border="1" style="color: black;">
        <tr> <th>Pending [{len(pending_uids)}]</th> <th>NAME</th> </tr>

    """
    htable1=''
    for pu in pending_uids:
        htable1+=f""" 
        <tr>
        <td><a href="{ url_for('route_storeuser', subpath=pu) }" target="_blank">{pu}</a></td>
        <td>{db[pu][2]}</td>
        </tr>
        """
    htable1+=f"""</table>
    <br>
    <table border="1" style="color: maroon;">
        <tr> <th>Absent [{len(absent_uids)}]</th><th>NAME</th> </tr>
    """
    htable2 = ''
    for pu in absent_uids:
        htable2+=f""" 
        <tr>
        <td>{pu}</td>
        <td>{db[pu][2]}</td>
        </tr>
        """
    htable2+=f"""</table>
    <br>
    <table border="1" style="color: black;">
        <tr>
            <th>Evaluated [{len(finished_uids)}]</th>
            <th>NAME</th>
            <th>SCORE</th>
            <th>REMARK</th>
            <th>EVALUATOR</th>
            
        </tr>
    """
    htable3 = ''
    #EVAL_ORD = ['UID', 'NAME', 'SCORE', 'REMARK', 'BY']
    counter = { k:0 for k in dbevaluatorset}
    for k in sorted(list(dbsub.keys())):
        v = dbsub[k]
        counter[v[4]]+=1
        htable3+=f"""
        <tr>
            <td><a href="{ url_for('route_storeuser', subpath=v[0]) }" target="_blank">{v[0]}</a></td>
            <td>{v[1]}</td>
            <td>{v[2]}</td>
            <td>{v[3]}</td>
            <td>{v[4]}</td>
        </tr>
        """
    htable3+=f"""</table><br>
    <h2>Evaluator Stats<h2>
    <table border="1" style="color: black;">
        <tr><th>Evaluator</th><th>#</th></tr>
    """
    htable4 = ''
    for k,v in counter.items():
        htable4+=f"""
        <tr>
            <td>{k} | {db[k][2]}</td>
            <td>{v}</td>
        </tr>
        """
    htable4+=f"""</table><br><hr></body></html>"""
    return render_template_string( htable0+htable1+htable2+htable3+htable4)

    

# ------------------------------------------------------------------------------------------
# eval
# ------------------------------------------------------------------------------------------
@app.route('/eval', methods =['GET', 'POST'], defaults={'req_uid': ''})
@app.route('/eval/<req_uid>')
def route_eval(req_uid):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    form = UploadFileForm()
    submitter = session['uid']
    results = []
    global db, dbsub
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
        if not req_uid:
            if ('+' in session['admind']) or ('X' in session['admind']):
                status, success = f"Eval Access is Enabled", True
            else: status, success = f"Eval Access is Disabled", False
        else:
            iseval = ('X' in session['admind']) or ('+' in session['admind'])
            if app.config['reeval']:
                if iseval:
                    in_uid = f'{req_uid}'
                    if in_uid: 
                    

                        in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
                        record = db.get(in_query, None)
                        if record is not None: 
                            erecord = dbsub.get(in_query, None)
                            if erecord is not None:
                                del dbsub[in_query]
                                dprint(f"▶ {session['uid']} ◦ {session['named']} has reset evaluation for {erecord[0]} ◦ {erecord[1]} (already evaluated by [{erecord[-1]}] with score [{erecord[2]}]) via {request.remote_addr}")
                                status, success =  f"Evaluation was reset for {record[1]} ◦ {record[2]}", True
                            else: status, success =  f"User {record[1]} ◦ {record[2]} has not been evaluated", False
                        else: status, success =  f"User '{in_query}' not found", False
                    else: status, success =  f"User-id was not provided", False
                else: status, success =  "You are not allow to reset evaluation", False
            else: status, success =  "Evaluation reset is disabled for this session", False
            if success: persist_subdb()


    return render_template('evaluate.html', success=success, status=status, form=form, results=results)

# ------------------------------------------------------------------------------------------


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
        if app.config['muc']==0 or app.config['disableupload']: 
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
        if session['uid'] in dbsub or app.config['disableupload']: return redirect(url_for('route_home'))

    folder_name = os.path.join( app.config['uploads'], session['uid']) 
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        for f in file_list: os.remove(os.path.join(folder_name, f))
        dprint(f'● {session["uid"]} ◦ {session["named"]} used purge via {request.remote_addr}')
        session['filed']=[]
    return redirect(url_for('route_home'))
# ------------------------------------------------------------------------------------------

class HConv: # html converter
    TEMPLATE = style.template_board
    NOSCRIPT = bool(parsed.nos)


    # @staticmethod
    # def convert(abs_path):
    #     new_abs_path = f'{abs_path}.html'
    #     if abs_path.lower().endswith(".ipynb"):
    #         try:
    #             x = __class__.nb2html( abs_path, __class__.TEMPLATE ,  __class__.NOSCRIPT )
    #             return True, x #(f"rendered Notebook to HTML @ {new_abs_path}")
    #         except: return False, (f"failed to rendered Notebook to HTML @ {new_abs_path}") 
    #     else: return False, (f"Cannot render this file as HTML: {os.path.basename(abs_path)}")

    @staticmethod
    def convertx(abs_path):
        new_abs_path = f'{abs_path}.html'
        if abs_path.lower().endswith(".ipynb"):
            try:
                x = __class__.nb2htmlx(abs_path, __class__.TEMPLATE, __class__.NOSCRIPT, 
                tlink=style.icon_gotop, dlink=style.icon_getfile, durl='#', align='left' )
                return True, x #(f"rendered Notebook to HTML @ {new_abs_path}")
            except: return False, (f"failed to rendered Notebook to HTML @ {new_abs_path}") 
        else: return False, (f"Cannot render this file as HTML: {os.path.basename(abs_path)}")

    @staticmethod
    def remove_tag(page, tag): # does not work on nested tags
        fstart, fstop = f'<{tag}', f'/{tag}>'
        while True:
            istart = page.find(fstart)
            if istart<0: break
            istop = page[istart:].find(fstop)
            page = f'{page[:istart]}{page[istart+istop+len(fstop):]}'
        return page
    
    # @staticmethod
    # def nb2html(source_notebook, template_name='lab', no_script=True, html_title=None, parsed_title='Notebook',):
    #     if html_title is None: # auto infer
    #         html_title = os.path.basename(source_notebook)
    #         iht = html_title.rfind('.')
    #         if not iht<0: html_title = html_title[:iht]
    #         if not html_title: html_title = (parsed_title if parsed_title else os.path.basename(os.path.dirname(source_notebook)))
    #     try:    
    #         page, _ = HTMLExporter(template_name=template_name).from_file(source_notebook,  dict(  metadata = dict( name = f'{html_title}' )    )) 
    #         if no_script: page = __class__.remove_tag(page, 'script') # force removing any scripts
    #     except: page = None
    #     return  page


        
    STYLE_ACTIONS = """
    .btn_header {
        background-color: #FFFFFF; 
        margin: 0px 0px 0px 6px;
        padding: 12px 6px 12px 6px;
        border-style: solid;
        border-width: thin;
        border-color: #000000;
        color: #000000;
        font-weight: bold;
        font-size: medium;
        border-radius: 5px;
    }

    .btn_actions {
        background-color: #FFFFFF; 
        padding: 2px 2px 2px 2px;
        margin: 5px 5px 5px 5px;
        border-style: solid;
        border-color: silver;
        border-width: thin;
        color: #000000;
        font-weight: bold;
        font-size: medium;
        border-radius: 2px;
    }


    """
    @staticmethod
    def nb2htmlx(source_notebook, 
                template_name='lab', no_script=False, 
                html_title=None, favicon=True, 
                header=1, tlink='top', dlink='download', durl='#', align='left'):
        # ==============================================================
        if html_title is None:
            html_title = os.path.basename(source_notebook)
            if html_title.lower().endswith(".ipynb"): html_title = html_title[:-6]
        page, _ = HTMLExporter(template_name=template_name) \
                .from_file(source_notebook, dict(metadata=dict(name = f'{html_title}')),) 
        soup = BeautifulSoup(page, 'html.parser')
        # ==============================================================
        
        if no_script:
            for script in soup.find_all('script'): script.decompose()  # Find all script tags and remove them
        
        if favicon:
            link_tag = soup.new_tag('link')
            link_tag['rel'] = 'icon'
            link_tag['href'] = 'favicon.ico'
            soup.head.insert(0, link_tag)

        #if tlink or hlink or dlink or header: 
        style_tag = soup.new_tag('style')
        style_tag['type'] = 'text/css'
        style_tag.string = __class__.STYLE_ACTIONS
        soup.head.insert(0, style_tag)


        #if hlink or dlink or header:
        ndiv = soup.new_tag('div')
        ndiv['align'] = f'{align}'
        html_string = ""
        #if auth: html_string += f'<a class="btn_actions" href="/logout">{llink}</a>'
        #if hlink: html_string += f'<a class="btn_actions" href="/">{hlink}</a>' 
        if dlink: html_string += f'<a class="btn_actions" href="{durl}">{dlink}</a>' 
        if header: html_string += f'<span class="btn_header">{html_title} @ ./{os.path.relpath(source_notebook, app.config["base"])}</span>'
        html_string += f'<br>'
        nstr = BeautifulSoup(html_string, 'html.parser')
        ndiv.append(nstr) 
        soup.body.insert(0, ndiv)

        if tlink:
            ndiv = soup.new_tag('div')
            ndiv['align'] = f'{align}'
            html_string = f'<hr><a class="btn_actions" href="#">{tlink}</a><br>'
            nstr = BeautifulSoup(html_string, 'html.parser')
            ndiv.append(nstr) 
            soup.body.append(ndiv)

        # ==============================================================
        # final_page = soup.prettify()
        # ==============================================================
        return soup.prettify()








# ------------------------------------------------------------------------------------------
# store
# ------------------------------------------------------------------------------------------
def list_store_dir(abs_path):
    dirs, files = [], []
    with os.scandir(abs_path) as it:
        for i,item in enumerate(it):
            if item.is_file(): files.append((i, item.name, item.name.startswith(".")))
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
                else: return f"Invalid args for store actions"
            

        if os.path.isdir(abs_path):
            if not request.args: 
                dirs, files = list_store_dir(abs_path)
                return render_template('store.html', dirs=dirs, files=files, subpath=subpath, form=form)
            else:
                if not can_admin: return "You cannot perform this action"
                if "." not in os.path.basename(abs_path) and os.path.abspath(abs_path)!=os.path.abspath(app.config['store']): #delete this dir
                    # if '!' in request.args:
                    #     try:
                    #         os.removedirs(abs_path)
                    #         dprint(f"● {session['uid']} ◦ {session['named']} deleted directory at {abs_path} # {subpath} via {request.remote_addr}")
                    #         return redirect(url_for('route_store', subpath=os.path.dirname(subpath)))
                    #     except:
                    #         return f"Error deleting the directory"
                    if '!' in request.args:
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
                elif ("html" in request.args): 
                    dprint(f"● {session['uid']} ◦ {session['named']} converting to html from {subpath} via {request.remote_addr}")
                    try:
                        hstatus, hmsg = HConv.convertx(abs_path)
                    except: hstatus, hmsg = False, f"Exception while converting notebook to web-page"
                    return hmsg
                    #dprint(f"{TABLINE}{'... ✓' if hstatus else '... ✗'} {hmsg}")
                    #return redirect(url_for('route_storeuser', subpath=os.path.dirname(subpath))) 
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
            try:
                hstatus, hmsg = HConv.convertx(abs_path)
            except: hstatus, hmsg = False, f"Exception while converting notebook to web-page"
            return hmsg
            #dprint(f"{TABLINE}{'... ✓' if hstatus else '... ✗'} {hmsg}")
            #return redirect(url_for('route_storeuser', subpath=os.path.dirname(subpath))) 
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

def toggle_upload():
    r""" disables uploads by setting app.config['disableupload']"""
    app.config['disableupload'] = not app.config['disableupload']

    
    if app.config['disableupload']: 
        STATUS, SUCCESS =  f"Uploads are now disabled", True
        dowhat = 'disabled'
    else: 
        STATUS, SUCCESS =  f"Uploads are now enabled", True
        dowhat = 'enabled'
    dprint(f"▶ {session['uid']} ◦ {session['named']} has {dowhat} uploads via {request.remote_addr}")
    return STATUS, SUCCESS 


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
                elif '~' in request.args: STATUS, SUCCESS = toggle_upload()
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

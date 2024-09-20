__doc__="""
Topics Server - a flask-based web app for sharing files.

This file can be run as stand alone as well. 
The only package dependencies are Flask, Flask-WTF, waitress (and optionally nbconvert)

The --dir represents the `WORKSPACE` directory
Run using the `--dir` option like 
    python __main__.py --dir=path/to/workspace
Here `workspace` folder will contain a `configs.py` file (will be created if not found)

The app will create 2 temporary folder - `templates` and `static`
These will be created in the same path where this script is placed (not the workspace folder)
The arguments --cos (create on start) abd --coe (clean on exit) can be used to manage these filess
"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------------------
import os, re, argparse, getpass, random, logging, importlib.util
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
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# args parsing
# ------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='', help="path of workspace directory")
parser.add_argument('--verbose', type=int, default=2, help="verbose level in logging")

parser.add_argument('--log', type=str, default='', help="path of log dir - keep blank to disable logging")
parser.add_argument('--logpre', type=str, default='', help="adds this to the start of logfile name (works when logging is enabled)")
parser.add_argument('--logname', type=str, default='%Y_%m_%d_%H_%M_%S_%f', help="name of logfile as formated string (works when logging is enabled)")
parser.add_argument('--logpost', type=str, default='', help="adds this to the end of logfile name (works when logging is enabled)")

parser.add_argument('--con', type=str, default='', help="config name - if not provided, uses 'default'")
parser.add_argument('--reg', type=str, default='', help="if specified, allow users to register with specified access string such as DABU or DABUS+")
parser.add_argument('--cos', type=int, default=1, help="use 1 to create-on-start - create (overwrites) pages")
parser.add_argument('--coe', type=int, default=1, help="use 1 to clean-on-exit - deletes pages")

parser.add_argument('--access', type=str, default='', help="if specified, allow users to add access string such as DABU or DABUS+")
parsed = parser.parse_args()
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
LOGDIR = f'{parsed.log}'                               # define log dir - contains all logs
LOGFILE = None
if LOGDIR and parsed.verbose>0: 

    LOGFILENAME = f'{parsed.logpre}{fnow(parsed.logname)}{parsed.logpost}'
    if not LOGFILENAME: exit(f'[!] Provided logfile nameLogging directory was not found and could not be created is blank!')

    try: os.makedirs(LOGDIR, exist_ok=True)
    except: exit(f'[!] Logging directory was not found and could not be created')
# ------------------------------------------------------------------------------------------
    try:
        # Set up logging to a file
        LOGFILE = os.path.join(LOGDIR, LOGFILENAME)
        logging.basicConfig(filename=LOGFILE, level=logging.INFO, format='%(asctime)s - %(message)s')
        # also output to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    except: exit(f'[!] Logging could not be setup at {LOGFILE}')
# ------------------------------------------------------------------------------------------


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

sprint(f'Starting...')

sprint(f'↪ Logging @ {LOGFILE}')

# ------------------------------------------------------------------------------------------
WORKDIR = f'{parsed.dir}'                               # define working dir - contains all bases
if not WORKDIR: WORKDIR = os.getcwd()                   # if still not specified, set as getcwd
try: os.makedirs(WORKDIR, exist_ok=True)
except: fexit(f'[!] Workspace directory was not found and could not be created')
sprint(f'↪ Workspace directory is {WORKDIR}')
#-----------------------------------------------------------------------------------------
# ==> read configurations
#-----------------------------------------------------------------------------------------
CONFIG = parsed.con if parsed.con else 'default' # the config-dict to read from
CONFIG_MODULE = 'configs'  # the name of configs module
CONFIGS_FILE = f'{CONFIG_MODULE}.py' # the name of configs file

CSV_DELIM=','
SSV_DELIM='\n'
MAX_STR_LEN=50
CSV_DTYPE=f'U{MAX_STR_LEN*2}'

LOGIN_ORD = ['ADMIN','UID','NAME','PASS']
SUBMIT_ORD = ['UID', 'NAME', 'SCORE', 'REMARK', 'BY']

DEFAULT_USER = 'admin'
DEFAULT_ACCESS = f'DABUSR+-'
"""
DEFAULT_ACCESS:
D   Read from [D]ownloads
A   Read from [A]rchives
B   Access [B]oard
U   Perform [U]pload
S   Read from [S]elf Uploads
R   Read from [R]eports
+   Admin access enabled
X   Reset access enabled (password reset)
-   Not included in submission
"""

#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# password policy
#-----------------------------------------------------------------------------------------
def VALIDATE_PASS(instr):   # a function that can validate the password - returns bool type
    try: assert (len(instr) < MAX_STR_LEN) and bool(re.fullmatch("(\w|@|\.)+", instr)) # alpha_numeric @.
    except AssertionError: return False
    return True
#-----------------------------------------------------------------------------------------
# uid policy
def VALIDATE_UID(instr):   # a function that can validate the uid - returns bool type
    try: assert (len(instr) < MAX_STR_LEN) and bool(re.fullmatch("(\w)+", instr)) # alpha_numeric 
    except AssertionError: return False
    return True
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# name policy
def VALIDATE_NAME(instr): return  (len(instr) >0) and (len(instr) < MAX_STR_LEN) and bool(re.fullmatch("((\w)(\w|\s)*(\w))|(\w)", instr)) # alpha-neumeric but no illegal spaces before or after
#-----------------------------------------------------------------------------------------

# this is useful for docker
# we try to read args from os env variables
def DEFAULT_CONFIG_GENERATE(): return """

def merged(a:dict, b:dict): return {**a, **b}

default = dict(    

    # -------------------------------------# general info
    topic        = "tOpIcS",               # topic text (main banner text)
    welcome      = "Welcome!",             # msg shown on login page
    register     = "Register!",            # msg shown on register (new-user) page
    emoji        = "◉",                    # emoji shown of login page and seperates uid - name
    rename       = 0,                      # if rename=1, allows users to update their names when logging in
    case         = 0,                      # case-sentivity level in uid
                                            #   (if case=0 uids are not converted           when matching in database)
                                            #   (if case>0 uids are converted to upper-case when matching in database)
                                            #   (if case<0 uids are converted to lower-case when matching in database)
    
    # -------------------------------------# validation
    ext          = "",                     # csv list of file-extensions that are allowed to be uploaded e.g., ext = "jpg,jpeg,png,txt" (keep blank to allow all extensions)
    required     = "",                     # csv list of file-names that are required to be uploaded e.g., required = "a.pdf,b.png,c.exe" (keep blank to allow all file-names)
    maxupcount   = -1,                     # maximum number of files that can be uploaded by a user (keep -1 for no limit and 0 to disable uploading)
    maxupsize    = "40GB",                 # maximum size of uploaded file (html_body_size)
    
    # -------------------------------------# server config
    maxconnect   = 50,                     # maximum number of connections allowed to the server
    threads      = 4,                      # no. of threads used by waitress server
    port         = "8080",                 # port
    host         = "0.0.0.0",              # ip

    # ------------------------------------# file and directory information
    base 		 = "__base__",            # the base directory 
    secret       = "__secret__.txt",      # flask app secret
    login        = "__login__.csv",       # login database
    submit       = "__submit__.csv",      # submission database - created if not existing - reloads if exists
    uploads      = "__uploads__",         # uploads folder (uploaded files by users go here)
    reports      = "__reports__",         # reports folder (personal user access files by users go here)
    downloads    = "__downloads__",       # downloads folder
    archives     = "__archives__",        # archives folder
    board        = "__board__.ipynb",     # board file

    # --------------------------------------# style dict
    style        = dict(                   
                        # -------------# labels
                        downloads_ =    'Downloads',
                        uploads_ =      'Uploads',
                        archives_ =     'Archives',
                        board_=         'Board',
                        admin_=         'Admin',
                        logout_=        'Logout',
                        login_=         'Login',
                        new_=           'Register',
                        submit_=        'Eval',
                        resetpass_=     'Reset',
                        report_=        'Report',

                        # -------------# icons 
                        icon_board =    '🔰',
                        icon_admin=     '⭐',
                        icon_login=     '🔒',
                        icon_new=       '👤',
                        icon_home=      '🔘',
                        icon_downloads= '📥',
                        icon_uploads=   '📤',
                        icon_archives=  '📦',
                        icon_submit=    '✴️',
                        icon_report=    '📜',

                        # -------------# admin actions 
                        aa_ref_downloads =  '📥',
                        aa_ref_archives=    '📦',
                        aa_db_write=     	'💾',
                        aa_db_read=       	'👁️‍🗨️',
                        aa_ref_board=      	'🔰',
                        aa_reset_pass= 		'🔑',

                        # -------------# board style ('lab'  'classic' 'reveal')
                        template_board = 'lab', 
                    )
    )

""" 


def DEFAULT_CONFIG_WRITE(file_path):
    with open(file_path, 'w', encoding='utf-8') as f: f.write(DEFAULT_CONFIG_GENERATE())



def DICT2CSV(path, d, ord):
    with open(path, 'w', encoding='utf-8') as f: 
        f.write(CSV_DELIM.join(ord)+SSV_DELIM)
        for v in d.values(): f.write(CSV_DELIM.join(v)+SSV_DELIM)

APPEND_ACCESS = f'{parsed.access}'.strip().upper()

def CSV2DICT(path, key_at):
    with open(path, 'r', encoding='utf-8') as f: 
        s = f.read()
        lines = s.split(SSV_DELIM)
        d = dict()
        for line in lines[1:]:
            if line:
                cells = line.split(CSV_DELIM)
                d[f'{cells[key_at]}'] = cells
        return d

def GET_SECRET_KEY(postfix):
    randx = lambda : random.randint(1111111111, 9999999999)
    r1 = randx()
    for _ in range(datetime.datetime.now().second): _ = randx()
    r2 = randx()
    for _ in range(datetime.datetime.now().second): _ = randx()
    r3 = randx()
    for _ in range(datetime.datetime.now().second): _ = randx()
    r4 = randx()
    return ':{}:{}:{}:{}:{}:'.format(r1,r2,r3,r4,postfix)


def GET_VALID_RE_PATTERN(validext):
    if not validext: return ".+"
    pattern=""
    for e in validext: pattern+=f'{e}|'
    return pattern[:-1]


def CREATE_LOGIN_FILE(login_xl_path):  
    this_user = getpass.getuser()
    if not (VALIDATE_UID(this_user)):  this_user=DEFAULT_USER
    DICT2CSV(login_xl_path, { f'{this_user}' : [DEFAULT_ACCESS,  f'{this_user}', f'{this_user}', f''] }, LOGIN_ORD ) # save updated login information to csv
    return this_user


def READ_DB_FROM_DISK(path, key_at):
    try:    return CSV2DICT(path, key_at), True
    except: return dict(), False
# ------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------
# Special Objects
#-----------------------------------------------------------------------------------------
class Fake:
    def __len__(self): return len(self.__dict__)
    def __init__(self, **kwargs) -> None:
        for name, attribute in kwargs.items():  setattr(self, name, attribute)
#-----------------------------------------------------------------------------------------

# ******************************************************************************************




#-----------------------------------------------------------------------------------------
# try to import configs
# inside the WORKDIR there should be 'configs.py' file
# check if 'configs.py` exsists or not`
CONFIGS_FILE_PATH = os.path.join(WORKDIR, CONFIGS_FILE) # should exsist under workdir
if not os.path.isfile(CONFIGS_FILE_PATH):
    sprint(f'↪ Creating default config "{CONFIGS_FILE}" ...')
    DEFAULT_CONFIG_WRITE(CONFIGS_FILE_PATH)
try: 
    # Load the module from the specified file path
    c_spec = importlib.util.spec_from_file_location(CONFIG_MODULE, CONFIGS_FILE_PATH)
    c_module = importlib.util.module_from_spec(c_spec)
    c_spec.loader.exec_module(c_module)
    sprint(f'↪ Imported config-module "{CONFIG_MODULE}" from {c_module.__file__}')
except: fexit(f'[!] Could import configs module "{CONFIG_MODULE}" at "{CONFIGS_FILE_PATH[:-3]}"')
try:
    sprint(f'↪ Reading config from {CONFIG_MODULE}.{CONFIG}')
    config_dict = getattr(c_module, CONFIG)
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
# ******************************************************************************************


    
# Read base dir first 
BASEDIR = os.path.abspath(((os.path.join(WORKDIR, args.base)) if args.base else WORKDIR))
try:     os.makedirs(BASEDIR, exist_ok=True)
except:  fexit(f'[!] base directory  @ {BASEDIR} was not found and could not be created') 
sprint(f'⚙ Base dicectiry: {BASEDIR}')
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
    this_user = CREATE_LOGIN_FILE(LOGIN_XL_PATH)
    sprint(f'⇒ Created new login with user "{this_user}" at file: {LOGIN_XL_PATH}')

# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# SUBMIT DATABASE - CSV
# ------------------------------------------------------------------------------------------
if not args.submit: SUBMIT_XL_PATH = None # fexit(f'[!] submission file was not provided!')    
else: SUBMIT_XL_PATH = os.path.join( BASEDIR, args.submit)
    

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
# archive settings
# ------------------------------------------------------------------------------------------
if not args.archives: fexit(f'[!] archives folder was not provided!')
ARCHIVE_FOLDER_PATH = os.path.join( BASEDIR, args.archives) 
try: os.makedirs(ARCHIVE_FOLDER_PATH, exist_ok=True)
except: fexit(f'[!] archives folder @ {ARCHIVE_FOLDER_PATH} was not found and could not be created')
sprint(f'⚙ Archive Folder: {ARCHIVE_FOLDER_PATH}')


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










ALLOWED_EXTENSIONS = set([x.strip() for x in args.ext.split(',') if x])  # a set or list of file extensions that are allowed to be uploaded 
if '' in ALLOWED_EXTENSIONS: ALLOWED_EXTENSIONS.remove('')
VALID_FILES_PATTERN = GET_VALID_RE_PATTERN(ALLOWED_EXTENSIONS)
REQUIRED_FILES = set([x.strip() for x in args.required.split(',') if x])  # a set or list of file extensions that are required to be uploaded 
if '' in REQUIRED_FILES: REQUIRED_FILES.remove('')
def VALIDATE_FILENAME(filename):   # a function that checks for valid file extensions based on ALLOWED_EXTENSIONS
    if '.' in filename: 
        name, ext = filename.rsplit('.', 1)
        safename = f'{name}.{ext.lower()}'
        if REQUIRED_FILES:  isvalid = (safename in REQUIRED_FILES)
        else:               isvalid = bool(re.fullmatch(f'.+\.({VALID_FILES_PATTERN})$', safename))
    else:               
        name, ext = filename, ''
        safename = f'{name}'
        if REQUIRED_FILES:  isvalid = (safename in REQUIRED_FILES)
        else:               isvalid = (not ALLOWED_EXTENSIONS)
    return isvalid, safename

def str2bytes(size):
    sizes = dict(KB=2**10, MB=2**20, GB=2**30, TB=2**40)
    return int(float(size[:-2])*sizes.get(size[-2:].upper(), 0))
MAX_UPLOAD_SIZE = str2bytes(args.maxupsize)     # maximum upload file size 
MAX_UPLOAD_COUNT = ( inf if args.maxupcount<0 else args.maxupcount )       # maximum number of files that can be uploaded by one user
INITIAL_UPLOAD_STATUS = []           # a list of notes to be displayed to the users about uploading files
if REQUIRED_FILES: INITIAL_UPLOAD_STATUS.append((-1, f'accepted files [{len(REQUIRED_FILES)}]: {REQUIRED_FILES}'))
else:
    if ALLOWED_EXTENSIONS:  INITIAL_UPLOAD_STATUS.append((-1, f'allowed extensions [{len(ALLOWED_EXTENSIONS)}]: {ALLOWED_EXTENSIONS}'))
INITIAL_UPLOAD_STATUS.append((-1, f'max upload size: {DISPLAY_SIZE_READABLE(MAX_UPLOAD_SIZE)}'))
if not (MAX_UPLOAD_COUNT is inf): INITIAL_UPLOAD_STATUS.append((-1, f'max upload count: {MAX_UPLOAD_COUNT}'))
sprint(f'⚙ Upload Settings ({len(INITIAL_UPLOAD_STATUS)})')
for s in INITIAL_UPLOAD_STATUS: sprint(f' ⇒ {s[1]}')
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# html pages
# ------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Create HTML
# ------------------------------------------------------------------------------------------
style = getattr(args, 'style')
CAPTION_DOWNLOADS =     style['downloads_']
CAPTION_UPLOADS =       style['uploads_'] 
CAPTION_ARCHIVES =      style['archives_']  
CAPTION_BOARD =         style['board_'] 
CAPTION_ADMIN =         style['admin_'] 
CAPTION_LOGOUT =        style['logout_'] 
CAPTION_LOGIN =         style['login_'] 
CAPTION_NEW =           style['new_'] 
CAPTION_SUBMIT =        style['submit_'] 
CAPTION_RESET_PASS =    style['resetpass_'] 
CAPTION_REPORT =        style['report_'] 

ICON_BOARD =            style['icon_board'] 
ICON_ADMIN =            style['icon_admin'] 
ICON_LOGIN =            style['icon_login'] 
ICON_NEW =              style['icon_new'] 
ICON_HOME =             style['icon_home'] 
ICON_DOWNLOADS =        style['icon_downloads'] 
ICON_UPLOADS =          style['icon_uploads'] 
ICON_ARCHIVES =         style['icon_archives'] 
ICON_SUBMIT =           style['icon_submit'] 
ICON_REPORT =           style['icon_report'] 

AA_REFD=                style['aa_ref_downloads'] 
AA_REFA=                style['aa_ref_archives']
AA_DBW=                 style['aa_db_write']
AA_DBR=                 style['aa_db_read']
AA_REFB=                style['aa_ref_board']
AA_RESETPASS=           style['aa_reset_pass']


TEMPLATE_BOARD =        style['template_board'] 


# ******************************************************************************************
HTML_TEMPLATES = dict(
# ******************************************************************************************
board="""""",
# ******************************************************************************************
submit = """
<html>
    <head>
        <meta charset="UTF-8">
        <title> """+f'{ICON_SUBMIT}'+""" {{ config.topic }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">  
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->

    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        <a href="{{ url_for('route_home') }}" class="btn_back">Back</a>
        <a href="{{ url_for('route_submit') }}" class="btn_refresh">Refresh</a>
        <br>
        <br>
        {% if success %}
        <span class="admin_mid" style="animation-name: fader_admin_success;">✓ {{ status }} </span>
        {% else %}
        <span class="admin_mid" style="animation-name: fader_admin_failed;">✗ {{ status }} </span>
        {% endif %}
        <br>

        
        <br>

        <form action="{{ url_for('route_submit') }}" method="post">
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
        </form>
        <br>
        <br>
        <form action="{{ url_for('route_submit') }}" method="post">
            <input id="resetpass" name="resetpass" type="text" style="width:100px;color: #9a0808; background: rgb(255, 171, 171)" placeholder="uid" class="txt_submit"/>
            <input type="submit" class="btn_purge_large" value="Reset Password"> 
        </form>        
    </div>

            
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    </body>
</html>
""",
# ******************************************************************************************
admin = """
<html>
    <head>
        <meta charset="UTF-8">
        <title> """+f'{ICON_ADMIN}'+""" {{ config.topic }} | {{ session.uid }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">					 
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    
    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        <a href="{{ url_for('route_home') }}" class="btn_back">Back</a>
        <a href="{{ url_for('route_adminpage') }}" class="btn_refresh">Refresh</a>
        <br>
        <br>
        {% if success %}
        <span class="admin_mid" style="animation-name: fader_admin_success;">✓ {{ status }} </span>
        {% else %}
        <span class="admin_mid" style="animation-name: fader_admin_failed;">✗ {{ status }} </span>
        {% endif %}
        <br>
        <br>
        {% if '+' in session.admind %}
        <a href="{{ url_for('route_adminpage',req_cmd='ref_downloads') }}" class="btn_admin_actions">"""+f'{AA_REFD}'+"""<span class="tooltiptext">Refresh Downloads</span></a> <!--Update download-list --!>
        <a href="{{ url_for('route_adminpage',req_cmd='ref_archives') }}" class="btn_admin_actions">"""+f'{AA_REFA}'+"""<span class="tooltiptext">Refresh Archives</span></a> <!--Update archive-list --!>
        <a href="{{ url_for('route_adminpage',req_cmd='db_write') }}" class="btn_admin_actions">"""+f'{AA_DBW}'+"""<span class="tooltiptext">Persist Database</span></a> <!--Persist login-database --!>
        <a href="{{ url_for('route_adminpage',req_cmd='db_read') }}" class="btn_admin_actions">"""+f'{AA_DBR}'+"""<span class="tooltiptext">Reload Database</span></a> <!--Reload login-database --!>
        <a href="{{ url_for('route_adminpage',req_cmd='ref_board') }}" class="btn_admin_actions">"""+f'{AA_REFB}'+"""<span class="tooltiptext">Refresh Board</span></a> <!--Refresh board --!>
        <button class="btn_admin_actions" onclick="confirm_repass()">"""+f'{AA_RESETPASS}'+"""<span class="tooltiptext">Reset Password</span></button>
        
            <script>
                function confirm_repass() {
                let res = prompt("Enter UID", ""); 
                if (res != null) {
                    location.href = "{{ url_for('route_repass',req_uid='::::') }}".replace("::::", res);
                    }
                }
            </script>
        {% endif %}
    </div>
            
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
        <title> """+f'{ICON_LOGIN}'+""" {{ config.topic }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">  
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
            {% if config.rename>1 %}
            <input id="emojid" name="emojid" type="text" placeholder={{ config.emoji }} class="txt_login_small"/>
            {% endif %}
            <br>
            {% endif %}
            <br>
            <input type="submit" class="btn_login" value=""" +f'"{CAPTION_LOGIN}"'+ """> 
            <br>
            <br>
        </form>
    </div>

    <!-- ---------------------------------------------------------->
    
    <div align="center">
    <div>
    <span style="font-size: xx-large;">{{ config.emoji }}</span>
    <br>
    {% if config.reg %}
    <a href="{{ url_for('route_new') }}" class="btn_board">""" + f'{CAPTION_NEW}' +"""</a>
    {% endif %}
    </div>
    <!-- <a href="https://emojipicker.com/" target="_blank" class="btn_login">...</a> -->
    <!--<div style="font-size:large"><a href="https://github.com/NelsonSharma/topics"  target="_blank"> 📤 📥 </a></div>-->
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
        <title> """+f'{ICON_NEW}'+""" {{ config.topic }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">  
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
            <input type="submit" class="btn_board" value=""" + f'"{CAPTION_NEW}"' +"""> 
            <br>
            <br>
            
        </form>
    </div>

    <!-- ---------------------------------------------------------->
    
    <div align="center">
    <div>
    <span style="font-size: xx-large;">{{ config.emoji }}</span>
    <br>
    <a href="{{ url_for('route_login') }}" class="btn_login">""" + f'{CAPTION_LOGIN}' +"""</a>
    
    </div>
    <!-- <a href="https://emojipicker.com/" target="_blank" class="btn_login">...</a> -->
    <!--<div style="font-size:large"><a href="https://github.com/NelsonSharma/topics"  target="_blank"> 📤 📥 </a></div>-->
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
        <title> """+f'{ICON_DOWNLOADS}'+""" {{ config.topic }} | {{ session.uid }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">           
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    
    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        <a href="{{ url_for('route_home') }}" class="btn_back">Back</a>
        <br>
        <br>
        <div class="files_status">"""+f'{CAPTION_DOWNLOADS}'+"""</div>
        <br>
        <div class="files_list_down">
            <ol>
            {% for file in config.dfl %}
            <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}" style="text-decoration: none; color: rgb(20, 20, 20);" >{{ file }}</a></li>
            <br>
            {% endfor %}
            </ol>
        </div>
        <br>
        <br>
    </div>

    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    </body>
</html>
""",
# ******************************************************************************************
archives = """
<html>
    <head>
        <meta charset="UTF-8">
        <title> """+f'{ICON_ARCHIVES}'+""" {{ config.topic }} | {{ session.uid }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">           
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    
    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        <a href="{{ url_for('route_home') }}" class="btn_back">Back</a>
        <br>
        <br>
        <div class="files_status">"""+f'{CAPTION_ARCHIVES}'+"""</div>
        <br>
        <div class="files_list_down">
            <ol>
            {% for file in config.afl %}
            <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}" style="text-decoration: none; color: rgb(20, 20, 20);" >{{ file }}</a></li>
            <br>
            {% endfor %}
            </ol>
        </div>
        <br>
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
        <title> """+f'{ICON_UPLOADS}'+""" {{ config.topic }} | {{ session.uid }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">           
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    
    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        <a href="{{ url_for('route_home') }}" class="btn_back">Back</a>
        <br>
        <br>
        <div class="files_status">"""+f'{CAPTION_UPLOADS}'+"""</div>
        <br>
        <div class="files_list_down">
            <ol>
            {% for file in session.filed %}
            <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}" style="text-decoration: none; color: rgb(20, 20, 20);" >{{ file }}</a></li>
            <br>
            {% endfor %}
            </ol>
        </div>
        <br>
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
        <title> """+f'{ICON_REPORT}'+""" {{ config.topic }} | {{ session.uid }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">           
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    
    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        <a href="{{ url_for('route_home') }}" class="btn_back">Back</a>
        <br>
        <br>
        <div class="files_status">"""+f'{CAPTION_REPORT}'+"""</div>
        <br>
        <div class="files_list_down">
            <ol>
            {% for file in session.reported %}
            <li><a href="{{ (request.path + '/' if request.path != '/' else '') + file }}"  target="_blank" style="text-decoration: none; color: rgb(20, 20, 20);" >{{ file }}</a></li>
            <br>
            {% endfor %}
            </ol>
        </div>
        <br>
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
        <title> """+f'{ICON_HOME}'+""" {{ config.topic }} | {{ session.uid }} </title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">					 
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->
    
    <div align="left" style="padding: 20px;">
        <div class="topic_mid">{{ config.topic }}</div>
        <div class="userword">{{session.uid}} {{ session.emojid }} {{session.named}}</div>
        <br>
        <a href="{{ url_for('route_logout') }}" class="btn_logout">"""+f'{CAPTION_LOGOUT}'+"""</a>
        {% if "S" in session.admind %}
        <a href="{{ url_for('route_uploads') }}" class="btn_upload">"""+f'{CAPTION_UPLOADS}'+"""</a>
        {% endif %}
        {% if "D" in session.admind %}
        <a href="{{ url_for('route_downloads') }}" class="btn_download">"""+f'{CAPTION_DOWNLOADS}'+"""</a>
        {% endif %}
        {% if "A" in session.admind %}
        <a href="{{ url_for('route_archives') }}" class="btn_archive">"""+f'{CAPTION_ARCHIVES}'+"""</a>
        {% endif %}
        {% if "B" in session.admind and config.board %}
        <a href="{{ url_for('route_board') }}" class="btn_board" target="_blank">"""+f'{CAPTION_BOARD}'+"""</a>
        {% endif %}
        {% if 'X' in session.admind or '+' in session.admind %}
        <a href="{{ url_for('route_submit') }}" class="btn_submit">"""+f'{CAPTION_SUBMIT}'+"""</a>
        {% endif %}
        {% if 'R' in session.admind %}
        <a href="{{ url_for('route_reports') }}" class="btn_report">"""+f'{CAPTION_REPORT}'+"""</a>
        {% endif %}
        
        {% if '+' in session.admind %}
        <a href="{{ url_for('route_adminpage') }}" class="btn_admin">"""+f'{CAPTION_ADMIN}'+"""</a>
        {% endif %}
        <br>
        <br>
        {% if config.muc!=0 and "U" in session.admind %}
                <div class="status">
                    <ol>
                    {% for s,f in status %}
                    {% if s %}
                    {% if s<0 %}
                    <li style="color: #ffffff;">{{ f }}</li>
                    {% else %}
                    <li style="color: #47ff6f;">{{ f }}</li>
                    {% endif %}
                    {% else %}
                    <li style="color: #ff6565;">{{ f }}</li>
                    {% endif %}
                    {% endfor %}
                    </ol>
                </div>
                <br>
                {% if submitted<1 %}
                <form method='POST' enctype='multipart/form-data'>
                    {{form.hidden_tag()}}
                    {{form.file()}}
                    {{form.submit()}}
                </form>
                {% else %}
                <div class="upword">Your Score is <span style="color:seagreen;">{{ score }}</span>  </div>
                {% endif %}
                <br>
                
                <div> <span class="upword">Uploads</span> 
                
                {% if "U" in session.admind and submitted<1 %}
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
# ******************************************************************************************
)
# ******************************************************************************************
CSS_TEMPLATES = dict(
# ****************************************************************************************** 0b7daa
style = """

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


.github_info {
    padding: 2px 10px;
    background-color: #516fa7; 
    color: #ffffff;
    font-size: medium;
    font-weight: bold;
    border-radius: 10px;
    font-family:monospace;
    text-decoration: none;
}


.topic{
    color: #000000;
    font-size: xxx-large;
    font-weight: bold;
    font-family:monospace;    
}


.msg_login{
    color: #060472; 
    font-size: large;
    font-weight: bold;
    font-family:monospace;    
    animation-duration: 3s; 
    animation-name: fader_msg;
}
@keyframes fader_msg {from {color: #ffffff;} to {color: #060472; } }


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

.topic_mid{
    color: #000000;
    font-size: x-large;
    font-style: italic;
    font-weight: bold;
    font-family:monospace;    
}

.userword{
    color: #12103c;
    font-weight: bold;
    font-family:monospace;    
    font-size: xxx-large;
}


.upword{
    color: #12103c;
    font-weight: bold;
    font-family:monospace;    
    font-size: xx-large;

}

.status{
    padding: 10px 10px;
    background-color: #232323; 
    color: #ffffff;
    font-size: medium;
    border-radius: 10px;
    font-family:monospace;
    text-decoration: none;
}


.files_status{
    font-weight: bold;
    font-size: x-large;
    font-family:monospace;
}

.files_list_up{
    padding: 10px 10px;
    background-color: #ececec; 
    color: #080000;
    font-size: medium;
    border-radius: 10px;
    font-family:monospace;
    text-decoration: none;
}

.files_list_down{
    padding: 10px 10px;
    background-color: #ececec; 
    color: #080000;
    font-size: x-large;
    font-weight: bold;
    border-radius: 10px;
    font-family:monospace;
    text-decoration: none;
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
.btn_admin {
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


.btn_admin_actions {
    padding: 2px 10px 2px;
    background-color: #FFFFFF; 
    border-style: solid;
    border-width: medium;
    border-color: #000000;
    color: #000000;
    font-weight: bold;
    font-size: xxx-large;
    border-radius: 5px;
    font-family:monospace;
    text-decoration: none;
}


.btn_admin_actions .tooltiptext {
  visibility: hidden;

  background-color: #000000;
  color: #ffffff;
  text-align: center;
  font-size: large;
  border-radius: 6px;
  padding: 5px 15px 5px 15px;

  position: absolute;
  z-index: 1;
}

.btn_admin_actions:hover .tooltiptext {
  visibility: visible;
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

.btn_archive{
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

.btn_back {
    padding: 2px 10px 2px;
    background-color: #a19636; 
    color: #FFFFFF;
    font-weight: bold;
    font-size: large;
    border-radius: 10px;
    font-family:monospace;
    text-decoration: none;
}


.admin_mid{
    color: #000000; 
    font-size: x-large;
    font-weight: bold;
    font-family:monospace;    
    animation-duration: 10s;
}
@keyframes fader_admin_failed {from {color: #ff0000;} to {color: #000000; } }
@keyframes fader_admin_success {from {color: #22ff00;} to {color: #000000; } }
"""
)
# ******************************************************************************************

PYDIR = os.path.dirname(__file__) # script directory of __main__.py
TEMPLATES_DIR, STATIC_DIR = os.path.join(PYDIR, "templates"), os.path.join(PYDIR, "static")

#sprint(f'↪ Creating html/css templates @ {PYDIR}')

os.makedirs(TEMPLATES_DIR, exist_ok=True)
#sprint(f'↪ Creating html templates @ {TEMPLATES_DIR}')
for k,v in HTML_TEMPLATES.items():
    h = os.path.join(TEMPLATES_DIR, f"{k}.html")
    if (not os.path.isfile(h)) or bool(parsed.cos):
        #sprint(f'  ↦ page:{h}')
        with open(h, 'w', encoding='utf-8') as f: f.write(v)

os.makedirs(STATIC_DIR, exist_ok=True)
#sprint(f'↪ Creating css templates @ {STATIC_DIR}')
for k,v in CSS_TEMPLATES.items():
    h = os.path.join(STATIC_DIR, f"{k}.css")
    if (not os.path.isfile(h)) or bool(parsed.cos):
        #sprint(f'  ↦ page:{h}')
        with open(h, 'w', encoding='utf-8') as f: f.write(v)

sprint(f'↪ Created html/css templates @ {PYDIR}')




# ------------------------------------------------------------------------------------------
BOARD_FILE_MD = None
BOARD_PAGE = ""
if args.board:
    try: 
        from nbconvert import HTMLExporter 
        has_nbconvert_package=True
    except:
        sprint(f'[!] Board will not be enabled since it requires nbconvert>=7.16.2 which is missing\n  ⇒ pip install nbconvert')
        has_nbconvert_package = False

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

if not BOARD_FILE_MD:   sprint(f'⚙ Board: Not Available')
else:                   sprint(f'⚙ Board: Is Available')
def update_board(): 
    global BOARD_PAGE
    res = False
    if BOARD_FILE_MD:
        try: 
            page,_ = HTMLExporter(template_name=TEMPLATE_BOARD).from_file(BOARD_FILE_MD, {'metadata':{'name':f'{ICON_BOARD} {CAPTION_BOARD} | {args.topic}'}}) 
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
# validation
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def read_logindb_from_disk():
    db_frame, res = READ_DB_FROM_DISK(LOGIN_XL_PATH, 1)
    if res: sprint(f'⇒ Loaded login file: {LOGIN_XL_PATH}')
    else: sprint(f'⇒ Failed reading login file: {LOGIN_XL_PATH}')
    return db_frame
def read_submitdb_from_disk():
    dbsub_frame = None
    if SUBMIT_XL_PATH: 
        dbsub_frame, ressub = READ_DB_FROM_DISK(SUBMIT_XL_PATH, 0)
        if ressub: sprint(f'⇒ Loaded submission file: {SUBMIT_XL_PATH}')
        else: sprint(f'⇒ Did not load submission file: [{SUBMIT_XL_PATH}] exists={os.path.exists(SUBMIT_XL_PATH)} isfile={os.path.isfile(SUBMIT_XL_PATH)}')
    return dbsub_frame
# ------------------------------------------------------------------------------------------
def write_logindb_to_disk(db_frame): # will change the order
    res = WRITE_DB_TO_DISK(LOGIN_XL_PATH, db_frame, LOGIN_ORD)
    if res: sprint(f'⇒ Persisted login file: {LOGIN_XL_PATH}')
    else:  sprint(f'⇒ PermissionError - {LOGIN_XL_PATH} might be open, close it first.')
    return res
def write_submitdb_to_disk(dbsub_frame, verbose=True): # will change the order
    ressub = True
    if SUBMIT_XL_PATH: 
        ressub = WRITE_DB_TO_DISK(SUBMIT_XL_PATH, dbsub_frame, SUBMIT_ORD)
        if verbose:
            if ressub: sprint(f'⇒ Persisted submission file: {SUBMIT_XL_PATH}')
            else:  sprint(f'⇒ PermissionError - {SUBMIT_XL_PATH} might be open, close it first.')
    return ressub
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
db =    read_logindb_from_disk()  #<----------- Created database here 
dbsub = read_submitdb_from_disk()  #<----------- Created database here 
sprint('↷ persisted submit-db [{}]'.format(write_submitdb_to_disk(dbsub)))

# = { k : [vu,  vn, 0.0, ''] for k,(va,vu,vn,_) in db.items() if '-' not in va} 
# -----------------------------------------------------------------------------------------
#print(dbsub)
def GetUserFiles(uid): 
    if not REQUIRED_FILES: return True # no files are required to be uploaded
    udir = os.path.join( app.config['uploads'], uid)
    has_udir = os.path.isdir(udir)
    if has_udir: return not (False in [os.path.isfile(os.path.join(udir, f)) for f in REQUIRED_FILES])
    else: return False

# ------------------------------------------------------------------------------------------
# application setting and instance
# ------------------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key =          APP_SECRET_KEY
app.config['base'] =      BASEDIR
app.config['uploads'] =   UPLOAD_FOLDER_PATH
app.config['reports'] =   REPORT_FOLDER_PATH
app.config['downloads'] = DOWNLOAD_FOLDER_PATH
app.config['archives'] =  ARCHIVE_FOLDER_PATH
app.config['emoji'] =     args.emoji
app.config['topic'] =     args.topic
app.config['dfl'] =       GET_FILE_LIST(DOWNLOAD_FOLDER_PATH)
app.config['afl'] =       GET_FILE_LIST(ARCHIVE_FOLDER_PATH)
app.config['rename'] =    int(args.rename)
app.config['muc'] =       MAX_UPLOAD_COUNT
app.config['board'] =     (BOARD_FILE_MD is not None)
app.config['reg'] =       (parsed.reg)
# ------------------------------------------------------------------------------------------
class UploadFileForm(FlaskForm): # The upload form using FlaskForm
    file = MultipleFileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")
# ------------------------------------------------------------------------------------------

#%% [4]
# app.route  > all app.route implemented here 
# ------------------------------------------------------------------------------------------
# login
# ------------------------------------------------------------------------------------------
@app.route('/', methods =['GET', 'POST'])
def route_login():
    LOGIN_NEED_TEXT =       '🔒'
    LOGIN_FAIL_TEXT =       '❌'     
    LOGIN_NEW_TEXT =        '🔥'
    LOGIN_CREATE_TEXT =     '🔑'    
    #NAME, PASS = 2, 3
    global db#, HAS_PENDING#<--- only when writing to global wariables
    if request.method == 'POST' and 'uid' in request.form and 'passwd' in request.form:
        in_uid = f"{request.form['uid']}"
        in_passwd = f"{request.form['passwd']}"
        in_name = f'{request.form["named"]}' if 'named' in request.form else ''
        in_emoji = f'{request.form["emojid"]}' if 'emojid' in request.form else app.config['emoji']
        if ((not in_emoji) or (app.config['rename']<2)): in_emoji = app.config['emoji']
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
                        #HAS_PENDING+=1
                        if in_name!=named and valid_name and (app.config['rename']>0) : 
                            db[uid][2]=in_name
                            #HAS_PENDING+=1
                            dprint(f'⇒ {uid} ◦ {named} updated name to "{in_name}" via {request.remote_addr}') 
                            named = in_name
                        else:
                            if in_name: dprint(f'⇒ {uid} ◦ {named} provided invalid name "{in_name}" (will not update)') 

                        warn = LOGIN_CREATE_TEXT
                        msg = f'[{in_uid}] ({named}) New password was created successfully'
                        dprint(f'● {in_uid} {in_emoji} {named} just joined via {request.remote_addr}')
           
                    else: # new password is invalid valid 
                        warn = LOGIN_NEW_TEXT
                        msg=f'[{in_uid}] New password is invalid - can use alpha-numeric, underscore and @-symbol'
                        
                                               
                else: #new password not provided                
                    warn = LOGIN_NEW_TEXT
                    msg = f'[{in_uid}] New password required - can use alpha-numeric, underscore and @-symbol'
                                           
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
                        session['admind'] = admind + APPEND_ACCESS
                        session['filed'] = os.listdir(folder_name)
                        session['reported'] = sorted(os.listdir(folder_report))
                        session['emojid'] = in_emoji 
                        
                        if in_name!=named and  valid_name and  (app.config['rename']>0): 
                            session['named'] = in_name
                            db[uid][2] = in_name
                            #HAS_PENDING+=1
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

@app.route('/new', methods =['GET', 'POST'])
def route_new():
    if not app.config['reg']: return "registration is not allowed"
    LOGIN_NEED_TEXT =       '👤'
    LOGIN_FAIL_TEXT =       '❌'     
    LOGIN_NEW_TEXT =        '🔥'
    LOGIN_CREATE_TEXT =     '🔑'    
    #NAME, PASS = 2, 3
    global db#, HAS_PENDING#<--- only when writing to global wariables
    if request.method == 'POST' and 'uid' in request.form and 'passwd' in request.form:
        in_uid = f"{request.form['uid']}"
        in_passwd = f"{request.form['passwd']}"
        in_name = f'{request.form["named"]}' if 'named' in request.form else ''
        in_emoji = f'{request.form["emojid"]}' if 'emojid' in request.form else app.config['emoji']
        if ((not in_emoji) or (app.config['rename']<2)): in_emoji = app.config['emoji']
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
                            msg=f'[{in_uid}] New password is invalid - can use alpha-numeric, underscore and @-symbol'
                            
                                                
                    else: #new password not provided                  
                        warn = LOGIN_NEW_TEXT
                        msg = f'[{in_uid}] New password required - can use alpha-numeric, underscore and @-symbol'
                                            

            else:
                warn, msg = LOGIN_FAIL_TEXT, f'[{in_uid}] is already registered' 

    else:
        if session.get('has_login', False):  return redirect(url_for('route_home'))
        msg = args.register
        warn = LOGIN_NEED_TEXT 
        
    return render_template('new.html', msg = msg,  warn = warn)

@app.route('/logout')
def route_logout():
    r""" logout a user and redirect to login page """
    if not session.get('has_login', False):  return redirect(url_for('route_login'))
    if not session.get('uid', False): return redirect(url_for('route_login'))
    if session['has_login']:  dprint(f'● {session["uid"]} {session["emojid"]} {session["named"]} has logged out via {request.remote_addr}') 
    else: dprint(f'✗ {session["uid"]} ◦ {session["named"]} was removed due to invalid uid ({session["uid"]}) via {request.remote_addr}') 
    # session['has_login'] = False
    # session['uid'] = ""
    # session['named'] = ""
    # session['emojid'] = ""
    # session['admind'] = ''
    # session['filed'] = []
    session.clear()
    return redirect(url_for('route_login'))
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# board
# ------------------------------------------------------------------------------------------
@app.route('/board', methods =['GET'])
def route_board():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'B' not in session['admind']:  return redirect(url_for('route_home'))
    return BOARD_PAGE

# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# archive
# ------------------------------------------------------------------------------------------
@app.route('/archives', methods =['GET'], defaults={'req_path': ''})
@app.route('/archives/<path:req_path>')
def route_archives(req_path):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if not 'A' in session['admind']: return redirect(url_for('route_home'))
    abs_path = os.path.join(app.config['archives'], req_path) # Joining the base and the requested path
    if not os.path.exists(abs_path): 
        dprint(f"⇒ requested file was not found {abs_path}") #Return 404 if path doesn't exist
        return abort(404) 
    if os.path.isfile(abs_path):  #(f"◦ sending file ")
        dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the file {req_path} via {request.remote_addr}')
        return send_file(abs_path) # Check if path is a file and serve
    return render_template('archives.html')
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# download
# ------------------------------------------------------------------------------------------
@app.route('/downloads', methods =['GET'], defaults={'req_path': ''})
@app.route('/downloads/<path:req_path>')
def route_downloads(req_path):
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if 'D' not in session['admind']:  return redirect(url_for('route_home'))
    abs_path = os.path.join(app.config['downloads'], req_path) # Joining the base and the requested path
    if not os.path.exists(abs_path): 
        dprint(f"⇒ requested file was not found {abs_path}") #Return 404 if path doesn't exist
        return abort(404) # (f"◦ requested file was not found") #Return 404 if path doesn't exist
    if os.path.isfile(abs_path):  #(f"◦ sending file ")
        dprint(f'● {session["uid"]} ◦ {session["named"]} just downloaded the file {req_path} via {request.remote_addr}')
        return send_file(abs_path) # Check if path is a file and serve
    return render_template('downloads.html')
# ------------------------------------------------------------------------------------------

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

@app.route('/submit', methods =['GET', 'POST'])
def route_submit():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    if request.method == 'POST': 
        global db, dbsub #, HAS_PENDING
        submitter = session['uid']
        if 'resetpass' in request.form:
            if ('X' in session['admind']) or ('+' in session['admind']):
                in_uid = f"{request.form['resetpass']}"
                if in_uid: 
                    in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
                    record = db.get(in_query, None)
                    if record is not None: 
                        admind, uid, named, _ = record
                        if (('X' not in admind) and ('+' not in admind)) or (submitter==uid):
                            db[uid][3]='' ## 3 for PASS  record['PASS'].values[0]=''
                            #HAS_PENDING+=1
                            dprint(f"▶ {submitter} ◦ {session['named']} just reset the password for {uid} ◦ {named} via {request.remote_addr}")
                            status, success =  f"Password was reset for {uid} {named}.", True
                        else: status, success =  f"You cannot reset password for account '{in_query}'.", False
                    else: status, success =  f"User '{in_query}' not found.", False
                else: status, success =  f"User-id was not provided.", False
            else: status, success =  "You are not allow to reset passwords.", False

        elif 'uid' in request.form and 'score' in request.form:
            if SUBMIT_XL_PATH:
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
                                    if scored[-1] == submitter or ('+' in session['admind']):
                                        if in_score:  dbsub[in_query][2] = in_score
                                        if in_remark: dbsub[in_query][3] = in_remark
                                        if in_score or in_remark : status, success =    f'Score/Remark Updated for [{in_uid}] {named}, current score is {dbsub[in_query][2]}. Remark is [{dbsub[in_query][3]}].', True
                                        else: status, success =                         f'Nothing was updated for [{in_uid}] {named}, current score is {dbsub[in_query][2]}. Remark is [{dbsub[in_query][3]}].', False
                                        dprint(f"▶ {submitter} ◦ {session['named']} updated the evaluation for {uid} ◦ {named} via {request.remote_addr}")
                                    else:
                                        status, success = f'[{in_uid}] {named} has been evaluated by [{scored[-1]}], you cannot update the information.', False
                                        dprint(f"▶ {submitter} ◦ {session['named']} is trying to revaluate {uid} ◦ {named} (already evaluated by [{scored[-1]}]) via {request.remote_addr}")
                
                else: status, success =  "You are not allow to evaluate.", False
            else: status, success =  "Evaluation is disabled.", False
        else: status, success = f"You posted nothing!", False
        
        if success: persist_subdb()
        
    else:
        if ('+' in session['admind']) or ('X' in session['admind']):
            status, success = f"Eval Access is Enabled", True
        else: status, success = f"Eval Access is Disabled", False
    
    return render_template('submit.html', success=success, status=status)



# ------------------------------------------------------------------------------------------
# home - upload
# ------------------------------------------------------------------------------------------
@app.route('/home', methods =['GET', 'POST'])
def route_home():
    if not session.get('has_login', False): return redirect(url_for('route_login'))
    form = UploadFileForm()
    folder_name = os.path.join( app.config['uploads'], session['uid']) 
    if SUBMIT_XL_PATH:
        submitted = int(session['uid'] in dbsub)
        score = dbsub[session['uid']][2] if submitted>0 else -1
    else: submitted, score = -1, -1

    if form.validate_on_submit() and ('U' in session['admind']):
        dprint(f"● {session['uid']} ◦ {session['named']} is trying to upload {len(form.file.data)} items via {request.remote_addr}")
        if app.config['muc']==0: 
            return render_template('home.html', submitted=submitted, score=score, form=form, status=[(0, f'✗ Uploads are disabled')])
        
        if SUBMIT_XL_PATH:
            if submitted>0: return render_template('home.html', submitted=submitted, score=score, form=form, status=[(0, f'✗ You have been evaluated - cannot upload new files for this session.')])

        result = []
        n_success = 0
        #---------------------------------------------------------------------------------
        for file in form.file.data:
            isvalid, sf = VALIDATE_FILENAME(secure_filename(file.filename))
        #---------------------------------------------------------------------------------
            
            if not isvalid:
                why_failed =  f"✗ File not accepted [{sf}] " if REQUIRED_FILES else f"✗ Extension is invalid [{sf}] "
                result.append((0, why_failed))
                continue

            file_name = os.path.join(folder_name, sf)
            if not os.path.exists(file_name):
                #file_list = os.listdir(folder_name)
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
    
    #file_list = session['filed'] #os.listdir(folder_name)
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
# if not '-' in session['admind']:        return redirect(url_for('route_home'))
# global dbsub
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
    if SUBMIT_XL_PATH:
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

 
# ------------------------------------------------------------------------------------------
# administrative
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
@app.route('/admin/', methods =['GET'], defaults={'req_cmd': ''})
@app.route('/admin/<req_cmd>')
def route_adminpage(req_cmd):
    r""" opens admin page """ 
    if not session.get('has_login', False): return redirect(url_for('route_login')) # "Not Allowed - Requires Login"
    in_cmd = f'{req_cmd}'
    if '+' in session['admind']: 
        if in_cmd: 
            if   in_cmd=="ref_downloads": STATUS, SUCCESS = update_dl()
            elif in_cmd=="ref_archives": STATUS, SUCCESS = update_al()
            elif in_cmd=="db_write": STATUS, SUCCESS = persist_db()
            elif in_cmd=="db_read": STATUS, SUCCESS = reload_db()
            elif in_cmd=="ref_board": STATUS, SUCCESS = refresh_board()
            else: STATUS, SUCCESS =  f"Invalid command '{in_cmd}'", False
        else: STATUS, SUCCESS =  f"Admin Access is Enabled", True
    else: 
        if in_cmd: STATUS, SUCCESS =  f"This action requires Admin access", False
        else:  STATUS, SUCCESS =  f"Admin Access is Disabled", False
    return render_template('admin.html',  status=STATUS, success=SUCCESS)

def update_dl():
    r""" refreshes the  downloads"""
    app.config['dfl'] = GET_FILE_LIST(DOWNLOAD_FOLDER_PATH)
    dprint(f"▶ {session['uid']} ◦ {session['named']} just refreshed the download list via {request.remote_addr}")
    return "Updated download-list", True #  STATUS, SUCCESS

def update_al():
    r""" refreshes the  downloads"""
    app.config['afl'] = GET_FILE_LIST(ARCHIVE_FOLDER_PATH)
    dprint(f"▶ {session['uid']} ◦ {session['named']} just refreshed the archive list via {request.remote_addr}")
    return "Updated archive-list", True #  STATUS, SUCCESS

def persist_db():
    r""" writes both dbs to disk """
    global db, dbsub
    if write_logindb_to_disk(db) and write_submitdb_to_disk(dbsub): #if write_db_to_disk(db, dbsub):
        dprint(f"▶ {session['uid']} ◦ {session['named']} just persisted the db to disk via {request.remote_addr}")
        STATUS, SUCCESS = "Persisted db to disk", True
    else: STATUS, SUCCESS =  f"Write error, file might be open", False
    return STATUS, SUCCESS 

def persist_subdb():
    r""" writes submit-db to disk """
    global dbsub
    if write_submitdb_to_disk(dbsub, verbose=False): 
        #dprint(f"▶ {session['uid']} ◦ {session['named']} just persisted the submit-db to disk via {request.remote_addr}")
        STATUS, SUCCESS = "Persisted db to disk", True
    else: STATUS, SUCCESS =  f"Write error, file might be open", False
    return STATUS, SUCCESS 

def reload_db():
    r""" reloads db from disk """
    global db, dbsub#, HAS_PENDING
    db = read_logindb_from_disk()
    dbsub = read_submitdb_from_disk()
    #HAS_PENDING=0
    dprint(f"▶ {session['uid']} ◦ {session['named']} just reloaded the db from disk via {request.remote_addr}")
    return "Reloaded db from disk", True #  STATUS, SUCCESS

def refresh_board():
    r""" refreshes the  board"""
    if update_board():
        dprint(f"▶ {session['uid']} ◦ {session['named']} just refreshed the board via {request.remote_addr}")
        return "Board was refreshed", True
    else: return "Board not enabled", False


# ------------------------------------------------------------------------------------------
# password reset
# ------------------------------------------------------------------------------------------
@app.route('/x/', methods =['GET'], defaults={'req_uid': ''})
@app.route('/x/<req_uid>')
def route_repass(req_uid):
    r""" reset user password"""
    if not session.get('has_login', False): return redirect(url_for('route_login')) # "Not Allowed - Requires Login"
    if ('+' in session['admind']): 
        in_uid = f'{req_uid}'
        if in_uid: 
            in_query = in_uid if not args.case else (in_uid.upper() if args.case>0 else in_uid.lower())
            global db#, HAS_PENDING
            record = db.get(in_query, None)
            if record is not None: 
                admind, uid, named, _ = record
                if ('+' not in admind) or (session['uid']==uid):
                    db[uid][3]='' ## 3 for PASS  record['PASS'].values[0]=''
                    #HAS_PENDING+=1
                    dprint(f"▶ {session['uid']} ◦ {session['named']} just reset the password for {uid} ◦ {named} via {request.remote_addr}")
                    STATUS, SUCCESS =  f"Password was reset for {uid} {named}", True
                else: STATUS, SUCCESS =  f"You cannot reset password for account '{in_query}'", False
            else: STATUS, SUCCESS =  f"User '{in_query}' not found", False
        else: STATUS, SUCCESS =  f"User-id was not provided", False
    else: STATUS, SUCCESS =  "You are not allow to reset passwords", False
    return render_template('admin.html',  status=STATUS, success=SUCCESS)
# ------------------------------------------------------------------------------------------



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#<-------------------DO NOT WRITE ANY NEW CODE AFTER THIS
endpoint = f'{args.host}:{args.port}' if args.host!='0.0.0.0' else f'localhost:{args.port}'
sprint(f'◉ http://{endpoint}')
start_time = datetime.datetime.now()
sprint('◉ start server @ [{}]'.format(start_time))
serve(app, # https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html
    host = args.host,          
    port = args.port,          
    url_scheme = 'http',     
    threads = args.threads,    
    connection_limit = args.maxconnect,
    max_request_body_size = MAX_UPLOAD_SIZE,
)
#<-------------------DO NOT WRITE ANY CODE AFTER THIS
end_time = datetime.datetime.now()
sprint('◉ stop server @ [{}]'.format(end_time))
sprint('↷ persisted login-db [{}]'.format(write_logindb_to_disk(db)))
sprint('↷ persisted submit-db [{}]'.format(write_submitdb_to_disk(dbsub)))

if bool(parsed.coe):
    sprint(f'↪ Cleaning up html/css templates...')
    try:
        for k,v in HTML_TEMPLATES.items():
            h = os.path.join(TEMPLATES_DIR, f"{k}.html")
            if  os.path.isfile(h) : os.remove(h)
        #sprint(f'↪ Removing css templates @ {STATIC_DIR}')
        for k,v in CSS_TEMPLATES.items():
            h = os.path.join(STATIC_DIR, f"{k}.css")
            if os.path.isfile(h): os.remove(h)
        os.removedirs(TEMPLATES_DIR)
        os.removedirs(STATIC_DIR)
        sprint(f'↪ Removed html/css templates @ {PYDIR}')
    except:
        sprint(f'↪ Could not remove html/css templates @ {PYDIR}')
sprint('◉ server up-time was [{}]'.format(end_time - start_time))
sprint(f'...Finished!')
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
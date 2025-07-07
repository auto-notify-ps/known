#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------
#%% Arg-parse
import os, argparse, datetime, logging
# python -m known.nbserver --help
parser = argparse.ArgumentParser()


parser.add_argument('--base',           type=str, default='',           help="path to base dir, defaults to current directory")
parser.add_argument('--template',       type=str, default='lab',        help="classic/[lab]/reveal")
parser.add_argument('--home',           type=str, default='',           help="home page if not specified, creates a new notebook with the same name as the base dir")
parser.add_argument('--noscript',       type=int, default=0,            help="[default=0] if =1, remove any embedded <script> tags (may break latex/mathjax)")
parser.add_argument('--nofiles',        type=int, default=0,            help="[default=0] if =1, prevents downloading files - only notebooks can be viewed and downloaded ")
parser.add_argument('--log',            type=str, default='',           help="log file name - keep empty for no logging")
parser.add_argument('--host',           type=str, default='0.0.0.0',    help="ip-Interface, keep 0.0.0.0 for all interfaces")
parser.add_argument('--port',           type=str, default='8080',       help="port")
parser.add_argument('--threads',        type=int, default=10,           help="no of threads used by waitress server")
parser.add_argument('--max_con',        type=int, default=500,          help="maximum number of connections allowed to the server")
parser.add_argument('--max_size',       type=str, default='1024MB',     help="maximum size of http body")
# Notebook decorations
parser.add_argument('--rtext',          type=str, default='♻️',         help="text for refresh link")      
parser.add_argument('--dtext',          type=str, default='📥️',         help="text for download link")       
parser.add_argument('--ttext',          type=str, default='🔝',         help="text for go-top link")
parser.add_argument('--htext',          type=str, default='🏠',         help="text for home link")
parser.add_argument('--ltext',          type=str, default='🔒',         help="text for logout link")
parser.add_argument('--halign',         type=str, default='left',       help="header alignment")
parser.add_argument('--header',         type=int, default=0,            help="shows text in the header")

# extra login
parser.add_argument('--login',        type=str, default='',           help="path to login-dir, each user has a file with password inside, keep blank to disbale authentication")
parser.add_argument('--secret',       type=str, default='',           help="the secret for flask, required only when login is enbaled")
parser.add_argument('--case',         type=int, default=0,            help="uid case sensetivity (-1 = to-lower) (0 = no-change) (1 = to-upper)")
parser.add_argument('--welcome',      type=str, default='Welcome',    help="welcome msg shown on the login page")
parser.add_argument('--https',        type=int, default=0,            help="if True, Tells waitress that its behind an nginx proxy - https://flask.palletsprojects.com/en/stable/deploying/nginx/")
parsed = parser.parse_args()

if parsed.login:
    LOGIN_XL_PATH = os.path.abspath(parsed.login)
    try:os.makedirs(LOGIN_XL_PATH, exist_ok=True)
    except:pass
    if not os.path.isdir(LOGIN_XL_PATH): exit(f'Login-dir not found at {LOGIN_XL_PATH}')
else: LOGIN_XL_PATH = None

#%% Logging
LOGFILE = f'{parsed.log}'
if LOGFILE: 
    try:
        logging.basicConfig(filename=LOGFILE, filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    except: exit(f'[!] Logging could not be setup at {LOGFILE}')
    def sprint(msg): logging.info(msg) 
else:
    def sprint(msg): print(msg) 

#%% imports
PROXY_FIX=bool(parsed.https)
import nbconvert, os
from bs4 import BeautifulSoup
from flask import Flask, request, abort, redirect, url_for, send_file, session, render_template_string
if PROXY_FIX: from werkzeug.middleware.proxy_fix import ProxyFix
from waitress import serve
#from flask import Flask, render_template, request, redirect, url_for, session, abort, send_file
#%% Common
str2bytes_sizes = dict(BB=2**0, KB=2**10, MB=2**20, GB=2**30, TB=2**40)
def str2bytes(size): return int(float(size[:-2])*str2bytes_sizes.get(size[-2:].upper(), 0))

style_actions = """
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
import re

def rematch(instr, pattern):  return \
    (len(instr) >= 0) and \
    (len(instr) <= 50) and \
    (re.match(pattern, instr))

def VALIDATE_PASS(instr):     return rematch(instr, r'^[a-zA-Z0-9~!@#$%^&*()_+{}<>?`\-=\[\].]+$')
def VALIDATE_UID(instr):      return rematch(instr, r'^[a-zA-Z0-9._@]+$') and instr[0]!="."


def get_passwd(dir, user):
    try:
        f = open(os.path.join(dir, user), 'r')
        passwd = f.read()
        f.close()
    except: passwd = None
    return passwd

def set_passwd(dir, user, passwd, create=False):
    try:
        p = os.path.join(dir, user)
        if not os.path.isfile(p): assert create
        f = open(p, 'w')
        f.write(passwd)
        f.close()
        return True
    except: return False

LOGIN_NEED_TEXT =       '👤' 
LOGIN_FAIL_TEXT =       '❌'     
LOGIN_NEW_TEXT =        '🔥'
LOGIN_CREATE_TEXT =     '🔑' 

login = """
<html>
    <head>
        <meta charset="UTF-8">
        <title> {{ config.title }} </title>
        <link rel="icon" href="/favicon.ico">
    </head>
    <body>
    <!-- ---------------------------------------------------------->
    </br>
    <!-- ---------------------------------------------------------->

    <div align="center">
        <br>
        <form action="{{ url_for('route_login') }}" method="post">
            <br>
            <div style="font-size: x-large;">{{ warn }}</div>
            <br>
            <div style="font-size: x-large; font-family: monospace">{{ msg }}</div>
            <br>
            <input id="uid" name="uid" type="text"  style="font-size: medium; font-family: monospace; text-align: center;" placeholder=" user-id " />
            <br>
            <br>
            <input id="passwd" name="passwd"   style="font-size: medium; font-family: monospace; text-align: center;" type="password" placeholder=" password " />
            <br>
            <br>
            <input type="submit" style="font-size: large;" value=""" + f'"{parsed.ltext}"' +"""> 
            <br>
            <br>
        </form>
    </div>

    <!-- ---------------------------------------------------------->
    
    <br>
    </div>
    <!-- ---------------------------------------------------------->
    </body>
</html>
"""

def nb2html(source_notebook, html_title=None, 
            template_name='lab', no_script=False, favicon=True, 
            auth=False, 
            llink='logout', 
            hlink='home', 
            header=0, 
            tlink='top', 
            clink='clear', curl='#',
            rlink='refresh', rurl='#', 
            dlink='download', durl='#', 
            align='left'):
    # ==============================================================
    if html_title is None:
        html_title = os.path.basename(source_notebook)
        if html_title.lower().endswith(".ipynb"): html_title = html_title[:-6]
    page, _ = nbconvert.HTMLExporter(template_name=template_name) \
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
    style_tag.string = style_actions
    soup.head.insert(0, style_tag)


    #if hlink or dlink or header:
    ndiv = soup.new_tag('div')
    ndiv['align'] = f'{align}'
    html_string = ""
    if auth: html_string += f'<a class="btn_actions" href="/logout">{llink}</a>'
    if hlink: html_string += f'<a class="btn_actions" href="/">{hlink}</a>' 
    if clink: html_string += f'<a class="btn_actions" href="{curl}">{clink}</a>' 
    if dlink: html_string += f'<a class="btn_actions" href="{durl}">{dlink}</a>' 
    if rlink: html_string += f'<a class="btn_actions" href="{rurl}">{rlink}</a>' 
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

def new_notebook(heading="", nbformat=4, nbformat_minor=2):
    return '{"cells": [{"cell_type": "markdown","metadata": {},"source": [ "# '+str(heading)+'" ] } ], "metadata": { }, "nbformat": '+str(nbformat)+', "nbformat_minor": '+str(nbformat_minor)+'}'

#%% App Setup 


BASE = os.path.abspath(parsed.base)
try: os.makedirs(BASE, exist_ok=True)
except: 
    sprint(f'No directory found at {BASE}, using current directory...')
    BASE = os.path.abspath(os.getcwd())

sprint(f'⇒ Serving from directory {BASE}')
EXT = ".ipynb"
PH = f'{parsed.home}'
if not PH: PH=os.path.basename(BASE)
HOME = f'{PH}{EXT}'
HOME_PATH = os.path.join(BASE, HOME)
if not os.path.isfile(HOME_PATH):
    try: 
        with open(HOME_PATH, 'w') as f: f.write(new_notebook(os.path.basename(BASE)))
    except: exit(f'The home page at {HOME_PATH} was not found and could not be created.')
if not os.path.isfile(HOME_PATH): exit(f'Home page "{HOME}" not found at {HOME_PATH}.')

AUTH_ON = (LOGIN_XL_PATH is not None)
SECRET_KEY = parsed.secret
if AUTH_ON and not SECRET_KEY: exit(f'Secret Key was not provided!')

app = Flask(__name__, static_folder=BASE, template_folder=BASE)
if PROXY_FIX: app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
if AUTH_ON: app.secret_key =  SECRET_KEY
app.config['auth'] = AUTH_ON
app.config['base'] = BASE
app.config['template'] = parsed.template
app.config['dtext'] = parsed.dtext
app.config['rtext'] = parsed.rtext
app.config['ttext'] = parsed.ttext
app.config['htext'] = parsed.htext
app.config['ltext'] = parsed.ltext
app.config['halign'] = parsed.halign
app.config['header'] = int(parsed.header)
app.config['home'] = HOME
app.config['title'] = os.path.basename(BASE)
app.config['ext'] = EXT
app.config['query_refresh'] = "!"
app.config['query_download'] = "?"
app.config['query_clear'] = "~"
app.config['no_script'] = bool(parsed.noscript)
app.config['no_files'] = bool(parsed.nofiles)

loaded_pages = dict()

#%% Routes Section
#from known import Verbose as vb
@app.route('/', methods =['GET'], defaults={'query': ''})
@app.route('/<path:query>')
def route_home(query):
    if app.config['auth']:
        if not session.get('has_login', False): return redirect(url_for('route_login'))
    #sprint("\n-------------------------------------------------------------")
    #sprint("[NEWREQUEST]")
    #sprint("-------------------------------------------------------------\n")
    #sprint(vb.show_(request))
    #sprint("\n-------------------------------------------------------------\n")
    refresh = app.config['query_refresh'] in request.args
    download = app.config['query_download'] in request.args
    clear = app.config['query_clear'] in request.args
    base, ext, home = app.config['base'], app.config['ext'], app.config['home']
    tosend = False
    
    if ('.' in os.path.basename(query)):    tosend = (not query.lower().endswith(ext))
    else:                                   query += ext #---> auto add extension
    if ext==query: 			                query=home

    showdlink = not((query==home) or (query==ext))
    #sprint (f'[{"🔸" if showdlink else "🔹"}]{request.remote_addr}\t[{request.method}] {request.url}')
    from_name = session["uid"] if app.config['auth'] else "👤"
    sprint (f'🔸 {from_name}\t({request.remote_addr})\t[{request.method}] {request.url}') # \n{request.headers}
    #sprint("__________________________________________________________________\n")
    requested = os.path.join(base, query) # Joining the base and the requested path
    if not ((os.path.isfile(requested)) and (not os.path.relpath(requested, base).startswith(base))): return abort(404)
    else:
        if tosend: 
            if requested.endswith("favicon.ico"): return send_file(requested, as_attachment=False) 
            else: return abort(403) if app.config['no_files'] else send_file(requested, as_attachment=False) 
        else:
            global loaded_pages
            if clear and not showdlink: # clear before loading
                loaded_pages.clear()
                return redirect(url_for('route_home'))
            if (requested not in loaded_pages) or refresh: loaded_pages[requested] = nb2html(
                    requested, 
                    html_title=app.config['title'] if not showdlink else None, 
                    template_name=app.config['template'], 
                    no_script=app.config['no_script'], 
                    favicon=True, 
                    auth = app.config['auth'],
                    llink=app.config['ltext'], 
                    tlink=app.config['ttext'], 
                    rlink=app.config['rtext'] if showdlink else None, 
                    dlink=app.config['dtext'] if showdlink else None, 
                    hlink = app.config['htext'] if showdlink else None,
                    clink = app.config['rtext'] if not showdlink else None,
                    header = app.config['header'] if showdlink else None,
                    durl=f"{request.path}?{app.config['query_download']}", 
                    rurl=f"{request.path}?{app.config['query_refresh']}", 
                    curl=f"{request.path}?{app.config['query_clear']}", 
                    align=app.config['halign'])
                #with open('??.html','w') as f: f.write(loaded_pages[requested]) # save a copy to disk?
            if refresh: return redirect(url_for('route_home', query=query))
            else:
                if download: return send_file(requested, as_attachment=True) #<--- downloading ipynbs
                else: return  loaded_pages[requested]


#%% TEMP ROUTES

# ------------------------------------------------------------------------------------------
# login
 
# ------------------------------------------------------------------------------------------
@app.route('/login', methods =['GET', 'POST'])
def route_login():
    if not app.config['auth']: 
        session['has_login'], session['uid'] = True, 'no-auth'
        return redirect(url_for('route_home'))
    if request.method == 'POST' and 'uid' in request.form and 'passwd' in request.form:
        in_uid = f"{request.form['uid']}"
        in_passwd = f"{request.form['passwd']}"

        uid = in_uid if not parsed.case else (in_uid.upper() if parsed.case>0 else in_uid.lower())
        valid_query = VALIDATE_UID(uid)
        if not valid_query : passwd = None
        else: passwd = get_passwd(LOGIN_XL_PATH, uid)
        if passwd is not None: 
            if not passwd: # fist login
                #warn = LOGIN_NEW_TEXT
                #msg = f'[{in_uid}] has not set their password'
                           
                if in_passwd: # new password provided
                    if VALIDATE_PASS(in_passwd): # new password is valid
                        set_passwd(LOGIN_XL_PATH, uid, in_passwd, create=False)
                        warn = LOGIN_CREATE_TEXT
                        msg = f'[{in_uid}] New password was created successfully'
                        sprint(f'● {in_uid} just joined via {request.remote_addr}')
           
                    else: # new password is invalid valid 
                        warn = LOGIN_NEW_TEXT
                        msg=f'[{in_uid}] New password is invalid - can use any of the alphabets (A-Z, a-z), numbers (0-9), underscore (_), dot (.) and at-symbol (@) only'
                        
                                               
                else: #new password not provided                
                    warn = LOGIN_NEW_TEXT
                    msg = f'[{in_uid}] New password required - can use any of the alphabets (A-Z, a-z), numbers (0-9), underscore (_), dot (.) and at-symbol (@) only'
                                                     
            else: # re login
                if in_passwd: # password provided 
                    if in_passwd==passwd:
                        session['has_login'], session['uid'] = True, uid
                        sprint(f'● {session["uid"]} has logged in via {request.remote_addr}') 
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
        msg = parsed.welcome
        warn = LOGIN_NEED_TEXT 
        
    return render_template_string(login, msg = msg,  warn = warn)
# ------------------------------------------------------------------------------------------

# logout
# ------------------------------------------------------------------------------------------
@app.route('/logout')
def route_logout():
    r""" logout a user and redirect to login page """
    if not app.config['auth']: return redirect(url_for('route_home'))
    if not session.get('has_login', False):  return redirect(url_for('route_login'))
    if not session.get('uid', False): return redirect(url_for('route_login'))
    if session['has_login']:  sprint(f'● {session["uid"]} has logged out via {request.remote_addr}') 
    else: sprint(f'✗ {session["uid"]} was removed due to invalid uid ({session["uid"]}) via {request.remote_addr}') 
    session.clear()
    return redirect(url_for('route_login'))
# ------------------------------------------------------------------------------------------

#%% Server Section
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

for endpoint in endpoints(parsed.host): sprint(f'◉ http://{endpoint}:{parsed.port}')
start_time = datetime.datetime.now()
sprint('◉ start server @ [{}]'.format(start_time))
serve(app,
    host = parsed.host,          
    port = parsed.port,          
    url_scheme = 'http',     
    threads = parsed.threads,    
    connection_limit = parsed.max_con,
    max_request_body_size = str2bytes(parsed.max_size),
    _quiet=True,
)
end_time = datetime.datetime.now()
sprint('◉ stop server @ [{}]'.format(end_time))
sprint('◉ server up-time was [{}]'.format(end_time - start_time))

#%%

# author: Nelson.S

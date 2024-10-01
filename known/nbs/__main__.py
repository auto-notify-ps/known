#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------

#%% Global

import os, argparse, datetime, logging
parser = argparse.ArgumentParser()
parser.add_argument('--base',           type=str, default='',                     help="path to base dir"     )
parser.add_argument('--template',       type=str, default='lab',                  help="classic/lab/reveal"   )
parser.add_argument('--title',          type=str, default='',                     help="to be used when empty title is found"   )

parser.add_argument('--query_refresh',  type=str, default='!',              help="# refresh ?!"         )
parser.add_argument('--query_download', type=str, default='?',              help="# download ??"        )
parser.add_argument('--no_script',      type=int, default=0,                help="if true, remove any embedded <script> tags")

parser.add_argument('--ext',            type=str, default='.ipynb',         help="extension for notebook files - case sensetive")
parser.add_argument('--log',            type=str, default='',               help="log file name")

parser.add_argument('--host',           type=str, default='0.0.0.0',                                    )
parser.add_argument('--port',           type=str, default='8088',                                       )
parser.add_argument('--threads',        type=int, default=50,                                           )
parser.add_argument('--max_connect',    type=int, default=500,                                          )
parser.add_argument('--max_size',       type=str, default='100MB',          help="size of http body"    )


parsed = parser.parse_args()

BASE = os.path.abspath(parsed.base)
if not os.path.isdir(BASE): exit(f'No directory found at {BASE}')


# ------------------------------------------------------------------------------------------
LOGFILE = f'{parsed.log}'
if LOGFILE: 
# ------------------------------------------------------------------------------------------
    try:
        # Set up logging to a file
        logging.basicConfig(filename=LOGFILE, filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s')
        # also output to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    except: exit(f'[!] Logging could not be setup at {LOGFILE}')
    def sprint(msg): logging.info(msg) 
# ------------------------------------------------------------------------------------------
else:
    def sprint(msg): print(msg) 
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
#%% Definitions

# import packages after all exit statements
from nbconvert import HTMLExporter 
from flask import Flask, request, abort, redirect, url_for, send_file
from waitress import serve

str2bytes_sizes = dict(BB=2**0, KB=2**10, MB=2**20, GB=2**30, TB=2**40)
def str2bytes(size): return int(float(size[:-2])*str2bytes_sizes.get(size[-2:].upper(), 0))

def remove_tag(page, tag): # does not work on nested tags
    fstart, fstop = f'<{tag}', f'/{tag}>'
    while True:
        istart = page.find(fstart)
        if istart<0: break
        istop = page[istart:].find(fstop)
        page = f'{page[:istart]}{page[istart+istop+len(fstop):]}'
    return page
    
def nb2html(source_notebook, template_name, no_script, html_title=None, parsed_title=None, dlink=''):
    if html_title is None: # auto infer
        html_title = os.path.basename(source_notebook)
        iht = html_title.rfind('.')
        if not iht<0: html_title = html_title[:iht]
        if not html_title: html_title = (parsed_title if parsed_title else os.path.basename(os.path.dirname(source_notebook)))
    try:    
        page, _ = HTMLExporter(template_name=template_name).from_file(source_notebook,  dict(  metadata = dict( name = f'{html_title}' )    )) 
        if no_script: page = remove_tag(page, 'script') # force removing any scripts
    except: page = None

    if dlink:
        fstart, fstop = f'<body', f'>'
        istart = page.find(fstart)
        if istart<0: return None
        istop = page[istart:].find(fstop)
        ins = f'<a href="{dlink}?{app.config["query_download"]}">⬇️</a>'
        page = f'{page[:istart+istop+len(fstop)]}{ins}{page[istart+istop+len(fstop):]}'


    return  page


#%% App Setup 

sprint(f'⇒ Serving from directory {BASE}')

app = Flask(__name__)
app.config['base'] = BASE
app.config['template'] = parsed.template
app.config['title'] = parsed.title
app.config['ext'] = parsed.ext # this is case sensetive
app.config['query_refresh'] = parsed.query_refresh 
app.config['query_download'] = parsed.query_download 
app.config['no_script'] = bool(parsed.no_script)
loaded_pages = dict()


#%% Routes Section

@app.route('/', methods =['GET'], defaults={'query': ''})
@app.route('/<path:query>')
def route_home(query):
    sprint (f'✴️ {request.remote_addr} {request.url} {request.args}')
    refresh = app.config['query_refresh'] in request.args
    download = app.config['query_download'] in request.args
    base, ext = app.config['base'], app.config['ext']
    tosend = False
    
    if ('.' in os.path.basename(query)):    tosend = (not query.lower().endswith(ext))
    else:                                   query += ext #---> auto add extension

    requested = os.path.join(base, query) # Joining the base and the requested path
    if not ((os.path.isfile(requested)) and (not os.path.relpath(requested, base).startswith(base))): return abort(404)
    else:
        if tosend: return send_file(requested)
        else:
            global loaded_pages
            if (requested not in loaded_pages) or refresh: loaded_pages[requested] = nb2html(requested, app.config['template'], app.config['no_script'],  html_title=None, parsed_title=app.config['title'], dlink=(f'{request.base_url}' if query!=ext else ''))
            return redirect(url_for('route_home', query=query)) if refresh else ( send_file(requested) if download else loaded_pages[requested])

#%% Server Section

endpoint = f'{parsed.host}:{parsed.port}' if parsed.host!='0.0.0.0' else f'localhost:{parsed.port}'
sprint(f'◉ http://{endpoint}')
start_time = datetime.datetime.now()
sprint('◉ start server @ [{}]'.format(start_time))
serve(app,
    host = parsed.host,          
    port = parsed.port,          
    url_scheme = 'http',     
    threads = parsed.threads,    
    connection_limit = parsed.max_connect,
    max_request_body_size = str2bytes(parsed.max_size),
)
end_time = datetime.datetime.now()
sprint('◉ stop server @ [{}]'.format(end_time))
sprint('◉ server up-time was [{}]'.format(end_time - start_time))

#%%

# author: Nelson.S

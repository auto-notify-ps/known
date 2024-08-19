#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------

#%% Global

import os, argparse, datetime
parser = argparse.ArgumentParser()
parser.add_argument('--base',           type=str, default='',               help="path to base dir"     )
parser.add_argument('--template',       type=str, default='lab',            help="classic/lab/reveal"   )
parser.add_argument('--host',           type=str, default='0.0.0.0',                                    )
parser.add_argument('--port',           type=str, default='8081',                                       )
parser.add_argument('--threads',        type=int, default=20,                                           )
parser.add_argument('--max_connect',    type=int, default=500,                                          )
parser.add_argument('--max_size',       type=str, default='100MB',          help="size of http body"    )
parsed = parser.parse_args()

BASE = os.path.abspath(parsed.base)
if not os.path.isdir(BASE): exit(f'No directory found at {BASE}')
print(f'⇒ Serving from directory {BASE}')

#%% Definitions

# import packages after all exit statements
from nbconvert import HTMLExporter 
from flask import Flask, request, abort, redirect, url_for, send_file
from waitress import serve

str2bytes_sizes = dict(BB=2**0, KB=2**10, MB=2**20, GB=2**30, TB=2**40)
def str2bytes(size): return int(float(size[:-2])*str2bytes_sizes.get(size[-2:].upper(), 0))

def nb2html(source_notebook, template_name, html_title=None):
    if html_title is None: html_title = os.path.basename(source_notebook)
    try:    page, _ = HTMLExporter(template_name=template_name).from_file(source_notebook, {'metadata':{'name':f'{html_title}'}}) 
    except: page = None
    return  page


#%% App Setup 

app = Flask(__name__)
app.config['base'] = BASE
app.config['ext'] = '.ipynb'
app.config['query_refresh'] = '!' # refresh ?!
app.config['query_download'] = '?' # download ??
loaded_pages = dict()


#%% Routes Section

@app.route('/', methods =['GET'], defaults={'query': ''})
@app.route('/<path:query>')
def route_home(query):
    print (f'✴️ {request.remote_addr} {request.url} {request.args}')
    refresh = app.config['query_refresh'] in request.args
    download = app.config['query_download'] in request.args
    base, ext = app.config['base'], app.config['ext']
    
    if not query.lower().endswith(ext): query += ext # allow only notebook files
    requested = os.path.join(base, query) # Joining the base and the requested path
    valid = not os.path.relpath(requested, base).startswith(base)
    exists = os.path.isfile(requested)
    if not (exists and valid): return abort(404)
    else:
        global loaded_pages
        if (requested not in loaded_pages) or refresh: loaded_pages[requested] = nb2html(requested, parsed.template,)
        return redirect(url_for('route_home', query=query)) if refresh else ( send_file(requested) if download else loaded_pages[requested])

#%% Server Section

endpoint = f'{parsed.host}:{parsed.port}' if parsed.host!='0.0.0.0' else f'localhost:{parsed.port}'
print(f'◉ http://{endpoint}')
start_time = datetime.datetime.now()
print('◉ start server @ [{}]'.format(start_time))
serve(app,
    host = parsed.host,          
    port = parsed.port,          
    url_scheme = 'http',     
    threads = parsed.threads,    
    connection_limit = parsed.max_connect,
    max_request_body_size = str2bytes(parsed.max_size),
)
end_time = datetime.datetime.now()
print('◉ stop server @ [{}]'.format(end_time))
print('◉ server up-time was [{}]'.format(end_time - start_time))

#%%

# author: Nelson.S

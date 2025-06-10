__doc__=r"""fuzz execute"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] Can not import {__name__}:{__file__}')
#-----------------------------------------------------------------------------------------

from bs4 import BeautifulSoup
import nbconvert, os, webbrowser
import tempfile


def nb2html(source_notebook, template_name, no_script):
    page, _ = nbconvert.HTMLExporter(template_name=template_name) \
            .from_file(source_notebook, dict(metadata=dict(name = f'{os.path.basename(source_notebook)}')),) 
    soup = BeautifulSoup(page, 'html.parser')
    # ==============================================================
    if no_script: # Find all script tags and remove them
        for script in soup.find_all('script'): script.decompose()  
    return soup.prettify()


#-----------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file',       type=str, default='',       help=f"(str) input notebook file (.ipynb)")
parser.add_argument('--template',   type=str, default='lab',    help="(str) nb template - classic , lab")
parser.add_argument('--script',     type=int, default=1,        help="(int) use 1 to enable scripts, required for math expressions")

parsed = parser.parse_args()
_file = f'{parsed.file}'
if not _file: exit(f'[!] Input file not provided')
_file = os.path.abspath(_file)
if not os.path.isfile(_file): exit(f'[!] Input file not found')
_template=f'{parsed.template}'
_no_script=not bool(parsed.script)
# # ---------------------------------------------------------------------------------
html_content=nb2html(source_notebook=_file, template_name=_template, no_script=_no_script)
# # ---------------------------------------------------------------------------------
file_path = os.path.join (tempfile.gettempdir(), f"{os.path.basename(_file)}.html")
with open(file_path, "w") as f:f.write(html_content)
webbrowser.open(f"file://{file_path}")
# # ---------------------------------------------------------------------------------

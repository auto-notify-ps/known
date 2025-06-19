__doc__=r"""notebook tools"""

__all__ = [ 
    'create', 
    'tohtml', 
    ]

from bs4 import BeautifulSoup
import nbconvert, os

def create(heading="# Notebook", nbformat=4, nbformat_minor=2, save=""): 
    r""" Create a new empty notebook and optionally saves it"""
    res = f"""
{{
    "cells": 
        [
            {{
                "cell_type": "markdown",
                "metadata": {{}},
                "source": [ "{heading}" ] 
            }} 
        ],
    "metadata": {{}}, 
    "nbformat": {nbformat}, 
    "nbformat_minor": {nbformat_minor}
}}
"""
    if save: 
        with open(save, 'w') as f: f.write(res)
    return res

def tohtml(source_notebook, template_name='lab', no_script=False):
    r""" Converts a notebook to html with added name (title) and template, optionally removes any scripts"""
    page, _ = nbconvert.HTMLExporter(template_name=template_name) \
            .from_file(source_notebook, dict(metadata=dict(name = f'{os.path.basename(source_notebook)}')),) 
    soup = BeautifulSoup(page, 'html.parser')
    if no_script: # Find all script tags and remove them
        for script in soup.find_all('script'): script.decompose()  
    return soup.prettify()




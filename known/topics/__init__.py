
__doc__=f""" 
-------------------------------------------------------------
topics - Flask-based web app for sharing files 
-------------------------------------------------------------
Usage:
python -m venv .venv
source .venv/bin/activate
python -m pip install Flask Flask-WTF waitress nbconvert
python -m pip install Flask==3.0.2 Flask-WTF==1.2.1 waitress==3.0.0 nbconvert==7.16.2
python topics.py 
-------------------------------------------------------------
Configs:
default `configs.py` file will be created
the dict named `current` will be choosen as the config
it should be defined at the end of the file
a config name `default` will be created 
it is used as a fall-back config
-------------------------------------------------------------
Note:
special string "::::" is used for replacing javascript on `repass` - uid and url should not contain this
special username 'None' is not allowed however words like 'none' will work
rename argument means (0 = not allowed) (1 = only rename) (2 = rename and remoji)
-------------------------------------------------------------


Note:
we use waitress for serving - which uses threads
threads in python are not really concurrent
all we need to do is declare variables as global when accessing from a thread
global variables are shared accross threads and not exclusive to thread
we must declare global variables only when writing to that variable but not on read 
"""




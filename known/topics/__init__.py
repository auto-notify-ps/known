
__doc__=f""" 
-------------------------------------------------------------
topics - Flask-based web app for sharing files 
-------------------------------------------------------------

Install the required packages
    
    python -m pip install Flask Flask-WTF waitress nbconvert

Note: the `nbconvert` is optional - without it, the 'board' functionality will be missing

To strat the server use,

    python -m known.topics --dir=path/to/workspace

To stop the server, press control-c or send keyboard interupt signal

Note: default `configs.py` file will be created under workspace directory
-> the dict named `current` will be choosen as the config - it should be defined at the end of the file
-> a config name `default` will be created - it is used as a fall-back config

Edit the configs file which should be located at `path/to/workspace/configs.py`
Multiple configs can be defined in that single file, only the `current` variable will be used

-------------------------------------------------------------
Note:
special string "::::" is used for replacing javascript on `repass` - uid and url should not contain this
special username 'None' is not allowed however words like 'none' will work
rename argument means (0 = not allowed) (1 = only rename) (2 = rename and remoji)
-------------------------------------------------------------
"""




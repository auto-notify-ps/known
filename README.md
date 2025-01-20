
# known

**known** is a collection of reusable python code.

## [1] Install from PyPI

```bash
python -m pip install known
```
The package is frequently updated by adding new functionality, make sure to have the latest version.
[Visit PyPI package homepage](https://pypi.org/project/known).


## [2] Install from GitHub

The github version is always upto-date. To install from github, use the following:
```bash
git clone https://github.com/auto-notify-ps/known.git
python -m pip install ./known
```
Cloned repo can be deleted after installation.

---
---
---
---
---
<br>

# known.fly

Flask based web app for sharing files and quiz evaluation

## Quickstart

* Install the required dependencies

```bash
python -m pip install Flask Flask-WTF waitress nbconvert 
```
* `nbconvert` package is *optional* - required only for the **Board** Page

* Start the server

```bash
python -m known.fly
```

* If the server was started for the first time (or config file was not found), a new config file `__configs__.py` will be created inside the **workspace directory**. It will contain the default configuration. In such a case the server will not start and the process is terminated with following output

```bash
⇒ Server will not start on this run, edit the config and start again
```

* One can edit this configuration file and start the server again. config file includes various options described as follows:

```python
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
```

* Additional Arguments can be passed while launching the server as follows:
```bash
python -m known.fly --help

usage: fly.py [-h] [--dir DIR] [--verbose VERBOSE] [--log LOG] [--con CON] [--reg REG] [--cos COS] [--coe COE] [--access ACCESS] [--msl MSL] [--eip EIP]

options:
  -h, --help         show this help message and exit
  --dir DIR          path of workspace directory [DEFAULT]: current diretory
  --verbose VERBOSE  verbose level in logging (0,1,2) [DEFAULT]: 2
  --log LOG          name of logfile as date-time-formated string, blank by default [Note: keep blank to disable logging]
  --con CON          config name (refers to a dict in __configs__.py - if not provided, uses 'default'
  --reg REG          if specified, allow users to register with that access string such as DABU or DABUS+
  --cos COS          use 1 to create-on-start - create (overwrites) pages [DEFAULT]: 1
  --coe COE          use 1 to clean-on-exit - deletes pages [DEFAULT]: 0
  --access ACCESS    if specified, adds extra premissions to access string for this session only
  --msl MSL          Max String Length for UID/NAME/PASSWORDS [DEFAULT]: 100
  --eip EIP          Evaluate Immediate Persis. If True (by-default), persist the eval-db after each single evaluation (eval-db in always persisted after update from template)
```

## Notes

* **Sessions** :
    * ShareFly uses only `http` protocol and not `https`. Sessions are managed on server-side. The location of the file containing the `secret` for flask app can be specified in the `__configs__.py` script. If not specified i.e., left blank, it will auto generate a random secret. Generating a random secret every time means that the users will not remain logged in if the server is restarted.

* **Database** :
    * The database of users is fully loaded and operated from RAM, therefore the memory usage depends on the number of registered users.
    * The offline database is stored in `csv` format and provides no security or ACID guarantees. The database is loaded when the server starts and is committed back to disk when the server stops. This means that if the app crashes, the changes in the database will not reflect. 
    * Admin users can manually **persist** (`!`) the database to disk and **reload** (`?`) it from the disk using the `/x/?` url.

* **Admin Commands** :
    * Admin users can issue commands through the `/x` route as follows:
        * Check admin access:        `/x`
        * Persist database to disk:  `/x?!`
        * Reload database from disk: `/x??`
        * Enable/Disable Uploads:    `/x?~`
        * Refresh Download List:     `/downloads??`
        * Refresh Board:             `/board??`

    * User-Related: 

        * Create a user with uid=`uid` and name=`uname`: 
            * `/x/uid?name=uname&access=DABU`
        * Reset Password for uid=`uid`:
            * `/x/uid`
        * Change name for uid=`uid`:
            * `/x/uid?name=new_name`
        * Change access for uid=`uid`:
            * `/x/uid?access=DABUSRX`
        

* **Access Levels** :
    * The access level of a user is specified as a string containing the following permissions:
        * `D`   Access Downloads
        * `A`   Access Store
        * `B`   Access Board
        * `U`   Perform Upload
        * `S`   Access Self Uploads
        * `R`   Access Reports
        * `X`   Eval access enabled
        * `-`   Not included in evaluation
        * `+`   Admin access enabled
    * The access string can contain multiple permissions and is specified in the `ADMIN` column of the `__login__.csv` file.

    * Note: Evaluators (with `X` access) cannot perform any admin actions except for resetting password through the `/x` url.

* **Store Actions** : `store/subpath?`
    * Create Folder : `store/subpath/my_folder??` (Only if not existing)
    * Delete Folder : `store/subpath/my_folder?!` (Recursive Delete)
    * Download File : `store/subpath/my_file?get`
    * Delete File   : `store/subpath/my_file?del`


* **App Routes** : All the `@app.route` are listed as follows:
    * Login-Page: `/`
    * Register-Page: `/new`
    * Logout and redirect to Login-Page: `/logout`
    * Home-Page: `/home`
    * Downloads-Page: `/downloads`
    * Reports-Page: `/reports`
    * Self-Uploads-Page: `/uploads`
    * Refresh Self-Uploads list and redirect to Home-Page: `/uploadf`
    * Delete all Self-Uploads and redirect to Home-Page: `/purge`
    * Store-Page (public): `/store`
    * User-Store-Page (evaluators): `/storeuser`
    * Enable/Disable hidden files in stores: `/hidden_show`
    * Evaluation-Page: `/eval`
    * Generate and Download a template for bulk evaluation: `/generate_eval_template`
    * Generate and View user reports: `/generate_submit_report`
    * Board-Page: `/board`
    * Admin-Access (redirects to Evalution-Page): `/x`


## Issue Tracking

#### [ 1 ] mistune version 3.1

* Reported: (Python3.10, ARM.aarch64)

* Error: The board file is not converted from `.ipynb` to `.html` even when `nbconvert` package is installed. 
```
AttributeError: 'MathBlockParser' object has no attribute 'parse_axt_heading'. Did you mean: 'parse_atx_heading'?
```

* Solution: use mistune version lower than `3.1` - find one at [PyPi](https://pypi.org/project/mistune/#history)
```bash
python -m pip install mistune==3.0.2
```

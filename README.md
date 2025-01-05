
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
    # -------------------------------------# general info
    topic        = "Fly",                  # topic text (main banner text)
    welcome      = "Login to Continue",    # msg shown on login page
    register     = "Register User",        # msg shown on register (new-user) page
    emoji        = "🦋",                   # emoji shown of login page and seperates uid - name
    rename       = 0,                      # if rename=1, allows users to update their names when logging in
    repass       = 1,                      # if repass=1, allows admins and evaluators to reset passwords for users - should be enabled in only one session (for multi-session)
    case         = 0,                      # case-sentivity level in uid
                                            #   (if case=0 uids are not converted when matching in database)
                                            #   (if case>0 uids are converted to upper-case when matching in database)
                                            #   (if case<0 uids are converted to lower-case when matching in database)
    
    # -------------------------------------# validation
    required     = "",                     # csv list of file-names that are required to be uploaded e.g., required = "a.pdf,b.png,c.exe" (keep blank to allow all file-names)
    extra        = 1,                      # if true, allows uploading extra file (other than required files)
    maxupcount   = -1,                     # maximum number of files that can be uploaded by a user (keep -1 for no limit and 0 to disable uploading)
    maxupsize    = "40GB",                 # maximum size of uploaded file (html_body_size)
    
    # -------------------------------------# server config
    maxconnect   = 50,                     # maximum number of connections allowed to the server
    threads      = 4,                      # no. of threads used by waitress server
    port         = "8888",                 # port
    host         = "0.0.0.0",              # ip (keep 0.0.0.0 for all interfaces)

    # ------------------------------------# file and directory information
    base         = "__base__",            # (auto create) the base directory - contains all other directories except html
    html         = "__pycache__",         # (auto create) use pycache dir to store flask html templates
    secret       = "secret.txt",      # (auto create) flask app secret is contained in this file
    login        = "login.csv",       # (auto create) login database in CSV format having four coloumns as (ADMIN, UID, NAME, PASS)
    eval         = "eval.csv",        # (auto create) evaluation database - created if not existing - reloads if exists
    uploads      = "uploads",         # (auto create) uploads folder (uploaded files by users go here)
    reports      = "reports",         # (auto create) reports folder (user read-only access files go here)
    downloads    = "downloads",       # (auto create) downloads folder (only files)
    store        = "store",           # (auto create) store folder (files and directory browsing)
    board        = "board.ipynb",     # (auto create) board file (a notebook file displayed as a web-page)
```

* Additional Arguments can be passed while launching the server as follows:
```python
# python -m known.fly --help
('--dir', type=str, default='', help="path of workspace directory, config file is located in here")
('--verbose', type=int, default=2, help="verbose level in logging")
('--log', type=str, default='', help="name of logfile as date-time-formated string e.g. fly_%Y_%m_%d_%H_%M_%S_%f_log.txt [Note: keep blank to disable logging]")
('--con', type=str, default='', help="config name - if not provided, uses 'default'")
('--reg', type=str, default='', help="if specified, allow users to register with specified access string such as DABU or DABUS+")
('--cos', type=int, default=1, help="use 1 to create-on-start - create (overwrites) pages")
('--coe', type=int, default=0, help="use 1 to clean-on-exit - deletes pages")
('--access', type=str, default='', help="if specified, adds extra premissions to access string for this session only")
('--msl', type=int, default=100, help="Max String Length for UID/NAME/PASSWORDS")
('--eip', type=int, default=1, help="Immediate Persists. If True, persist the eval-db after each single evaluation (eval-db in always persisted after update from template)")
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

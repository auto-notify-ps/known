
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
cd known
python -m pip install .
```
Cloned repo can be deleted after installation.

---
---
---

# 📦 Modules

---
---
<br>


# 🧩 known.Mailer

* Send emails from within your python code.

* Note: Implements sending functionality only, cannot read emails.

Usage:

```
known.Mailer.Mail(
    username = "sender@gmail.com",     # sender's userid - must be @gmail 
    password = "aaa bbb ccc ddd",      # sender's password - use app password
    Subject = "this is the subject line", 
    To = "person1@domain1.com,person2@domain2.com",  # csv list of email addresses
    Cc = "person3@domain3.com,person4@domain4.com",  # csv list of email addresses
    Body = "this is the body of the message", 
    Attached = ['attach.txt', '/home/attach.pdf'], # python list of attached files
    )

```

---
---
<br>


# 🧩 known.pix

Basic Image manupulation from the command line, 

```
python -m knwon.pix --help

options:
  -h, --help         show this help message and exit
  --action ACTION    (str) one of the static-methods inside the Actions class, 
                           can be - ['new', 'crop', 'extend', 'flip', 'rotate', 'convert', 'autoconvert']
  --args ARGS        (str) csv args accepted by the specified action - each action takes different args
  --input INPUT      (str) input image-file or a text/json-file containing multiple input image-file names
  --output OUTPUT    (str) output image-file or a text/json-file containing multiple output image-file names
  --files FILES      (str) multiple input image-file names - for custom action -- works only with --io=linux
  --io IO            (str) can be 'text' or 'json' or 'linux' - keep blank to io as 'image' - used if providing 
                           input/output file-names in a text/json file
  --verbose VERBOSE  (int) verbose level - 0 or 1

```

## Functionality

#### 1. Create a new image

* creates new images of given size and color 

* args is (int) 7-tuple (height, width, channels, blue, green, red, alpha)
    * `--args=h,w,c,b,g,r,a`

* fill-color (blue, green, red, alpha) depends on the specified number of channels
    * `--args=h,w,4,b,g,r,a`
    * `--args=h,w,3,b,g,r`
    * `--args=h,w,1,i`                  

Example - creating 32 x 64 image with 4 channels named 'new.jpg'
```
python -m known.pix --action=new --args=32,64,4,100,150,200,170 --output=new.jpg
```

#### 2. Crop an image

* crops an image using bounding box (y, x, h, w) 

* args is (int) 4-tuple (y-cord, x-coord, height, width) indicating a bounding box
    * `--args=y,x,h,w`

Example - 
```
python -m known.pix --action=crop --args=8,16,16,32 --input=new.jpg --output=cropped.jpg
```

#### 3. Extend an image

* extends an image using boundary distance 

* args is (int) 8-tuple (north, south, east, west, blue, green, red, alpha)
    * `--args=n,s,e,w,b,g,r,a`

* fill-color (blue, green, red, alpha) depends on the specified number of channels
    * `--args=n,s,e,w,b,g,r,a`
    * `--args=n,s,e,w,b,g,r`    
    * `--args=n,s,e,w,i`

Example - 
```
python -m known.pix --action=extend --args=10,5,6,12,123,123,0,100 --input=new.jpg --output=extended.jpg
```


#### 4. Flip an image

* flip an image (horizontally, vertically)

* args is (int) 2-tuple (horizontally, vertically) 

    * Flip horizontally          `--args=1,0`
    * Flip vertically            `--args=0,1`
    * Flip corners               `--args=1,1`
    * Flip nothing               `--args=0,0`

Example - 
```
python -m known.pix --action=flip --args=1,1 --input=extended.jpg --output=flipped.jpg
```

#### 5. Rotate an image

* rotate an image (clockwise or couter-clockwise)

* args is (int) 1-tuple (clockwise) 
    
    * Rotate clockwise               `--args=1`
    * Rotate counter-clockwise       `--args=0`

Example - 
```
python -m known.pix --action=rotate --args=1 --input=flipped.jpg --output=rotated.jpg
```

#### 6. Convert an image to a format

* converts between image formats *(as per output)*

* args is not used, target file type is infered from the output file extension

Example - 
```
python -m known.pix --action=convert --input=rotated.jpg --output=rotated.png
```

#### 7. Convert an image to multiple formats

* converts between image formats *(as per args)*

* args is (str) n-tuple specifying the extensions to be converted to

* output filenames are not used, the file-names are taken from input files

* the extensions are added as specified in args

* e.g., Convert png to jpg and webp      `--input=input.png --args=jpg,webp`
the output files will be `input.png` and `input.webp`

Example - 
```
python -m known.pix --action=autoconvert --input=rotated.png --args=jpeg,webp
```

---
---
<br>

# 🧩 known.fly

Flask based web app for sharing files and quiz evaluation

## Quickstart

* Install the required dependencies

```bash
python -m pip install Flask Flask-WTF waitress nbconvert 
```

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
    extra        = 1,                      # if true, allows uploading extra file (other than required)
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
    secret       = "secret.txt",      # file containing flask app secret (keep blank to generate random secret every time)
    login        = "login.csv",       # login database having four cols ADMIN, UID, NAME, PASS
    eval         = "eval.csv",        # evaluation database - created if not existing - reloads if exists
    uploads      = "uploads",         # uploads folder (uploaded files by users go here)
    reports      = "reports",         # reports folder (read-only files that are private to a user go here)
    downloads    = "downloads",       # downloads folder (public read-only access)
    store        = "store",           # store folder (public read-only, evaluators can upload and delete files)
    board        = "board.ipynb",     # board file (public read-only, a notebook displayed as a web-page)
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
    * `known.fly` uses only `http` protocol and not `https`. Sessions are managed on server-side. The location of the file containing the `secret` for flask app can be specified in the `__configs__.py` script. If not specified i.e., left blank, it will auto generate a random secret. Generating a random secret every time means that the users will not remain logged in if the server is restarted.

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
    * The access string can contain multiple permissions and is specified in the `ADMIN` column of the `login.csv` file.

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

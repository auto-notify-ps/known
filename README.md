
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
   
## Notes

* **Sessions** :
    * ShareFly uses only `http` protocol and not `https`. Sessions are managed on server-side. The location of the file containing the `secret` for flask app can be specified in the `__configs__.py` script. If not specified i.e., left blank, it will auto generate a random secret. Generating a random secret every time means that the users will not remain logged in if the server is restarted.

    * To enable `https`, one can generate a self-signed certificate and use `nginx` reverse proxy (edit `/etc/nginx/nginx.conf`)

      ```bash
         http {
            include       mime.types;
            default_type  application/octet-stream;
            sendfile        on;
            keepalive_timeout  65;
            types_hash_max_size 4096;
            types_hash_bucket_size 256;
            # HTTPS server
            server {
               # listen to default https port
               listen       443 ssl; 
               server_name  localhost;
               
               # provide self-signed certificates and private key here
               ssl_certificate      /path/to/certificate;
               ssl_certificate_key  /path/to/private-key;
               ssl_protocols        TLSv1.2 TLSv1.3;
               ssl_ciphers          HIGH:!aNULL:!MD5;
               ssl_prefer_server_ciphers    on;

            # set the proxy   
            location / {
               client_max_body_size 102400m; # (0 = no limit)
               proxy_pass http://127.0.0.1:8888; # map to http server running on loopback
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            }}
      ```


* **Database** :
    * The database of users is fully loaded and operated from RAM, therefore the memory usage depends on the number of registered users.
    * The offline database is stored in `csv` format and provides no security or ACID guarantees. The database is loaded when the server starts and is committed back to disk when the server stops. This means that if the app crashes, the changes in the database will not reflect. 
    * Admin users can manually **persist** (`!`) the database to disk and **reload** (`?`) it from the disk using the `/x/?` url.

* **Admin Commands** :
    * Admin users can issue commands through the `/x` route as follows:
        * Check admin access:        `/x`
        * Persist database to disk:  `/x/?!`
        * Reload database from disk: `/x/??`
        * Refresh Download List:     `/downloads/??`
        * Refresh Board:             `/board/??`

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
    * Delete Folder : `store/subpath/my_folder?!` (Only if empty `os.removedirs`)
    * Delete Folder : `store/subpath/my_folder?x` (Recursive Delete `shutils.rmtree`)
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



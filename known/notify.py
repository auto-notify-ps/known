#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] Can not import {__name__}:{__file__}')
#-----------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
from .mail import Mailer
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def SendPreHTML(body_str, body_file, attach_file, csvto, csvcc, subject, username, password):
    if body_str: body = str(body_str)
    elif body_file:
        with open(body_file, 'r') as f: body = f.read()
    else: body = ""
    if not csvto: csvto=username
    return Mailer.SendMail(
        username=username,  password=password,  Subject=subject, To=csvto, Cc=csvcc, 
        Body=f'<html><head></head><body><pre>{body}</pre></body></html>', Attached=attach_file,
        html=True, url='smtp.gmail.com', port='587', tls=True, )

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default='', help="sender profile (file-path) - keep blank to create one")

# if to-adress no provided, uses username (self-send)
parser.add_argument('--t', type=str, default='', help="to-addresses (csv-str) - keep blank to send to self (sender)")

# either provide m or b (m preceeds over b)
parser.add_argument('--m', type=str, default='', help="body - message (str) - either provide --m or --b")
parser.add_argument('--b', type=str, default='', help="body - file (path) - will be used only if --m not specified")

parser.add_argument('--s', type=str, default='', help="subject of email (str)")
parser.add_argument('--c', type=str, default='', help="cc-addresses (csv-str)")

parser.add_argument('--a', type=str, nargs='*',  default=[], help="attachments - zero or more file paths")

parsed = parser.parse_args()

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if not parsed.p: 
    import getpass
    #sp = input(f'Sender profile not provided,  ')
    print(f'Sender profile not provided ... ')
    u = input("Enter sender gmail-address ").strip()
    p = getpass.getpass("Enter sender gmail-password ")

    spname = input("Do you want to save these credentials? \n ... enter a profile path to save credentials or leave blank to continue without saving ... ").strip()
    
    if spname:
        sp = os.path.abspath(spname)
        with open(sp, 'w') as f:
            f.write(f'{u}\n{p}\n')
        print(f'Created new profile [{os.path.basename(sp)}] at "{sp}"\nUse it with --p argument\n')

else:
    with open(parsed.p, 'r') as f:  u, p = [l.strip() for l in f.readlines()][0:2]

try:
    sent = Mailer.SendPreHTML(
        body_str=parsed.m, 
        body_file=parsed.b, 
        attach_file=parsed.a, 
        csvto=parsed.t, 
        csvcc=parsed.c, 
        subject=parsed.s,
        username=u,
        password=p,
        )
    if sent: print(f'✅ Success')
    else:    print(f'❌ Failed')
except Exception as e: print(f'❗ Exception Occured\n{e}')

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

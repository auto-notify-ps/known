

class SCRIPTSCLASS: ...
SCRIPTS = SCRIPTSCLASS()

SCRIPTS.blacklist = """#!/bin/bash

# Usage

# Place the script in a folder and run with "new" argument to create "conf" and "service" files
# <> new

# add or remove IP addresses or ranges using "add" and "del" arguments
# <> add 0.0.0.0/0
# <> del 0.0.0.0/0

# restore "conf" usually at the startup (thats what "service" file is for)
# <> res

# --------------------------------------------
if [[ "$USER" != "root" ]]; then
    echo "requires sudo"
    exit 127
fi
# --------------------------------------------

# --------------------------------------------
if [[ -z "$1" ]]; then
    echo "[Arg #1 - Action] must be:"
    echo " -- [add] or [del] if adding/deleting IP Ranges to/from list"
    echo " -- [res] if restoring list"
    echo " -- [new] for creating new list"
    echo " -- [ls] for listing"
    exit 127
fi
# --------------------------------------------

# --------------------------------------------
SCRIPT_NAME="${BASH_SOURCE##*/}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# --------------------------------------------
ListName="$SCRIPT_NAME"
SourceFolder="$SCRIPT_DIR"
ScriptFile="$SourceFolder/$ListName"
ConfFile="$SourceFolder/$ListName.conf"
ServiceFile="$SourceFolder/$ListName.service"
# --------------------------------------------
echo "ListName: $ListName"
echo "SourceFolder: $SourceFolder"
echo "ScriptFile: $ScriptFile"
echo "ConfFile: $ConfFile"
echo "ServiceFile: $ServiceFile"
echo "User: $USER"
# --------------------------------------------




# --------------------------------------------
if [[ "$1" = "new" ]]; then
# --------------------------------------------    


    ipset create $ListName hash:net -exist
    ipset list $ListName
    ipset save $ListName > $ConfFile

    # create a service file too
    echo " ... press enter (blank) to create a service file at $ServiceFile ... "
    read choice
    if [[ -z "$choice" ]]; then
        echo "creating service file ... "

cat << EOF > "$ServiceFile"
[Unit]
Description=$ListName-Service
After=network-pre.target
Before=network.target
Wants=network-pre.target

[Service]
Type=oneshot
User=root
ExecStart=$ScriptFile res
Restart=no
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
EOF

        echo " ... press enter (blank) to link ... "
        read choice2
        if [[ -z "$choice2" ]]; then
            Spath='/etc/systemd/system'
            while :; do
                echo "linking to $Spath/$ListName.service, enter new path to change or leave blank to confirm"
                read choice3
                if ! [[ -z "$choice3" ]]; then
                    Spath="$choice3"
                else
                    break
                fi
            done
            ln -s $ServiceFile $Spath/$ListName.service
            systemctl daemon-reexec && systemctl daemon-reload
            sleep 1
            systemctl status $ListName
            echo " ... press enter (blank) to enable service ... "
            read choice4
            if [[ -z "$choice4" ]]; then
                systemctl enable $ListName
                sleep 1
                systemctl status $ListName
            fi
            
        else
            echo "not linking ..."
        fi
        
    else
        echo "not creating service file ..."
    fi

# --------------------------------------------
elif [[ "$1" = "ls" ]]; then 
# --------------------------------------------


    ipset list $ListName


# --------------------------------------------
elif [[ "$1" = "res" ]]; then 
# --------------------------------------------


    ipset restore < $ConfFile
    iptables -t raw -C PREROUTING -m set --match-set $ListName src -j DROP 2>/dev/null || \
    iptables -t raw -I PREROUTING -m set --match-set $ListName src -j DROP

    # extra rules

    # reject uid=1000 from using wifi
    #iptables -A OUTPUT -o wlan0 -m owner --uid-owner 1000 -j REJECT
    #ip6tables -A OUTPUT -o wlan0 -m owner --uid-owner 1000 -j REJECT

    # reject uid=1002 from using LAN
    #iptables -A OUTPUT -o eno1 -m owner --uid-owner 1002 -j REJECT
    #ip6tables -A OUTPUT -o eno1 -m owner --uid-owner 1002 -j REJECT


# --------------------------------------------
else
# --------------------------------------------


    if [[ -z "$2" ]]; then
        echo "[Arg #2] must be an one or more IPRanges if adding/deleting IP Ranges"
        exit 127
    fi
    
    args=("$@")
    Excluded=("${args[@]:1}")
    for xip in "${Excluded[@]}"; do
        echo "$xip"
        ipset $1 $ListName $xip -exist 
        #ss -K dst $xip
        #sudo ss -K dst <IP_ADDRESS> dport = <PORT>
    done
    ipset save $ListName > $ConfFile
    

# --------------------------------------------
fi
# --------------------------------------------

"""

SCRIPTS.py3install="""#!/bin/bash
# -----------------------------------------------
# Download and Install python from source code
# -----------------------------------------------

# This script will download python source code and install using make altinstall
# Download this script as <exe> and give execute permission
# chmod 777 ./<exe>

# make sure you have dependencies
# sudo apt install wget build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev libnss3-dev

# usage ./<exe> <major_version> <minor_version> <tarxz_location>

# e.g if you want to install Python-3.11.8 from internet
# major_version = 11
# minor_version = 8
# keep tarxz_location blank
# usage: ./<exe> 11 8

# e.g if you want to install Python-3.11.8 from local Python-3.11.8.tar.xz file
# major_version = 11
# minor_version = 8
# tarxz_location = /path/containing/Python-3.11.8.tar.xz
# supposing Python-3.11.8.tar.xz is in the Downloads folder then
# usage: ./<exe> 11 8 /home/$USER/Downloads
# NOTE: don't provide the full path of tar file, only provide the folder that it is inside
#		will automatically look for file named "Python-3.11.8.tar.xz" 
#		file name should not be changed from what it is by default

# Will try to clean up extracterd files, do not not run under any directory where python is going to be installed
# prefer home directory or downloads directory
# best way is to cd into Downloads folder and use 
# usage: ./<exe> 11 8 ./    # if you have the source code tar.xz
# usage: ./<exe> 11 8    # if you want to download and install from source code tar.xz
if [[ "$1" == '' ]] ; then 
	echo "major version required!"
	exit 1
fi
if [[ "$2" == '' ]] ; then
	echo "minor version required!"
	exit 2
fi
pyver=3.$1.$2
echo "Installing Python-$pyver"
echo "Installing python..."
if [[ "$3" == '' ]] ; then
	pyurl=https://www.python.org/ftp/python/$pyver/Python-$pyver.tar.xz
	echo "Downloading Python from $pyurl"
	wget $pyurl -O ./Python-$pyver.tar.xz
	tar -xf ./Python-$pyver.tar.xz --directory=./
else
	echo "Using Local Source from $3"
	ls -lash $3/Python-$pyver.tar.xz
	tar -xf $3/Python-$pyver.tar.xz --directory=./
fi
cd Python-$pyver
pwd
echo "Configuring..."
./configure --enable-optimizations --enable-loadable-sqlite-extensions --with-ensurepip=install
echo "Installing..."
sudo make -j $(nproc)
sudo make altinstall
cd ..
pwd
pyv=3.$1
echo "Testing..."
#"/usr/local/bin/python$pyv" --version
python$pyv --version
python$pyv -m pip --version
python$pyv -m pip list
python$pyv -c "import _sqlite3"
echo "Finished!"
echo "Cleaning up... ./Python-$pyver"
sudo rm -r ./Python-$pyver
echo "Finished installing Python-$pyver"
whereis python$pyv
echo "Done! python command is python$pyv"
"""

SCRIPTS.addservice="""#!/bin/bash

# --------------------------------------------
if [[ "$USER" != "root" ]]; then
    echo "requires sudo"
    exit 127
fi
# --------------------------------------------


Spath='/etc/systemd/system'

if [[ -z "$1" ]]; then
    echo "Service File (.service file's full path)"
    read ServiceFile
else
    ServiceFile="$1"
fi

if [[ -z "$ServiceFile" ]]; then
    echo "Source File not provided!"
    exit 2
fi

SourceFile=$(realpath "$ServiceFile")
echo "Source File is $SourceFile"
if ! [ -f "$SourceFile" ]; then
    echo "Source File not found!"
    exit 3
fi

SourceName=$(basename "$SourceFile")
Sname="${SourceName%.*}"
Sext="${SourceName##*.}"
DestName="$Sname"

echo "Source = $SourceName = ($Sname)+($Sext)"
echo "Dest = $DestName"

#echo "Destination Name (service name only - dont append .service)"
#read DestName
if [[ -z "$DestName" ]]; then
    echo "Dest not provided!"
    exit 2
fi


echo " ... press enter (blank) to link otherwise copy ... "
read choice
if [[ -z "$choice" ]]; then
    echo "linking..."
    ln -s $SourceFile $Spath/$DestName.service
else
    echo "copying..."
    cp -p $SourceFile $Spath/$DestName.service
fi

echo " ... press enter (blank) to daemon-reexec and daemon-reload ... "
read choice
if [[ -z "$choice" ]]; then
    systemctl daemon-reexec
    systemctl daemon-reload
    sleep 1
    systemctl status $DestName
fi
# ================================================================
echo " ... [#] done!"
# ================================================================
"""

SCRIPTS.alias="""

alias ipa='ip -c -br a && ip route'

alias ctl='sudo systemctl'
alias ctl-='sudo systemctl stop'
alias ctl+='sudo systemctl start'
alias ctl.='sudo systemctl status'
alias ctls='sudo systemctl --type=service'
alias services='sudo systemctl --type=service --state=running'
alias ctlx='sudo systemctl daemon-reexec'
alias ctlr='sudo systemctl daemon-reload'
alias ctlre='sudo systemctl daemon-reexec && sudo systemctl daemon-reload'
alias uctl='systemctl --user'
alias uctl-='systemctl --user stop'
alias uctl+='systemctl --user start'
alias uctl.='systemctl --user status'
alias uctls='systemctl --user --type=service'
alias uservices='systemctl --user --type=service --state=running'

alias ports='sudo ss -tuln'
alias portsp='sudo ss -tulnp'
alias conns='sudo ss -tuna'
alias connsd='sudo ss -tuna state connected'
alias connsp='sudo iptables -t raw -L PREROUTING -v -n'

alias jctl='sudo journalctl -u'
alias re1='sudo shutdown -r +1'
alias shut1='sudo shutdown -P +1'

alias secret='python3 -c "import secrets; print(secrets.token_urlsafe(64))"'
alias bt='btop --utf-force'
alias ff='fastfetch'

"""

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help="name of the script")
    parser.add_argument('--path', type=str, default='', help="path to create script, keep blank to ue the same name")
    parsed = parser.parse_args()


    if not parsed.name: exit(f'name not provided, use --name argument')
    else:
        if not hasattr(SCRIPTS, parsed.name): 
            available_names = "\n".join(list(SCRIPTS.__dict__.keys()))
            exit(f'name {parsed.name} does not exist. Available names are\n{available_names}\n')

    script_name = parsed.path if parsed.path else parsed.name
    script_path = os.path.abspath(script_name)
    if os.path.isdir(script_path): exit(f'save path is a directory')


    script_string = getattr(SCRIPTS, parsed.name)
    with open(script_path, 'w') as f: f.write(script_string)
    print(f'Script saved at {script_path}')
    print(f'Make it executable using\n chmod +x {script_path}')
    print(f'Use as a source \n source {script_path}')


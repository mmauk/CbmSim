#!/usr/bin/bash

: << 'END_COMMENT'
file: build_remote.sh
author: Sean Gallogly
Year: 2022

Usage: ./build_remote.sh [ OPTIONS ] 

Purpose: This script syncs a local version of the big simulation into
         the Cerebellum server within the Mauk Lab. It then attempts to
		 compile the synced version of the big simulation on Cerebellum,
		 logging standard outpout and any errors to the designated log files


Requirements: vpnui
              rsync

Additional Notes: This file assumes you have access to a private key which is
                  accepted by the public key on Cerebellum. There are various
				  resources to do this, but 

END_COMMENT

set -e

CLIENT_NAME=$USER
REMOTE_HOST=$(cat ~/.ssh/config | head -n 1 | awk '{print $2}')
REMOTE_USER=$(grep "[U|u]ser" ~/.ssh/config | awk '{print $2}')

CLIENT_PROJ_PATH="$HOME/Dev/Git/UT_Austin/Big_Sim/"
REMOTE_PROJ_PATH="/home/seang/Dev/Git/Big_Sim/"
LOG_FILE="$CLIENT_PROJ_PATH"logs/log_remote_sign_in_error

printf "Checking if vpn service is running\n"

if $(ps -e | grep -q "vpnui"); then
	printf "vpn service already running\n"
else	
	printf "opening vpn service...\n"
    # TODO: find a vpncli	
	({ vpnui 2>/dev/null; } &) 	
fi

printf "\nCommiting and pushing to remote...\n"
git add .
read -p "Enter your commit message: " commit_message
git commit -m "$commit_message"
git push origin master

printf "\nAttempting to log in via ssh to $REMOTE_USER@$REMOTE_HOST...\n\n"
ssh -T -E $LOG_FILE $REMOTE_USER@$REMOTE_HOST << 'EOL'
	cd /home/seang/Dev/Git/Big_Sim
	git pull origin master
	read -p "(b)uild or (c)lean? " command
	case $command in
		[bB])
			printf "Building...\n" && make ;;
		[cC])
			printf "Cleaning...\n" && make clean ;;
		*)
			printf "Command not recognized. Exiting\n" && exit 1 ;;
	esac
EOL 

printf "\n Finished with all tasks. Exiting..."

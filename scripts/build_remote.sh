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

read -p "Would you like to push the project to the remote? (y/n) " git_integrate 

[ "$git_integrate" != y ] && [ "$git_integrate" != n ] && \
	printf "input not recognized. Exiting\n" && exit 1

if [ "$git_integrate" = "y" ]; then
	printf "\nCommiting and pushing to remote...\n"
	git add .
	read -p "Enter your commit message: " commit_message
	git commit -m "$commit_message"
	git push origin master
fi

command=$1
printf "\nAttempting to log in via ssh to $REMOTE_USER@$REMOTE_HOST...\n\n"
ssh -T -E $LOG_FILE $REMOTE_USER@$REMOTE_HOST 'bash -s' < build_remote_helper.sh "$command $git_integrate"

printf "\nFinished with all tasks. Exiting...\n"


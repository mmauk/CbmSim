#!/usr/bin/bash

: << 'END_COMMENT'
file: remote_task.sh
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

declare -a git_push=( push to )
declare -a git_pull=( pull from )

CLIENT_NAME=$USER
REMOTE_HOST=$(cat ~/.ssh/config | head -n 1 | awk '{print $2}')
REMOTE_USER=$(grep "[U|u]ser" ~/.ssh/config | awk '{print $2}')

CLIENT_PROJ_PATH="$HOME/Dev/Git/UT_Austin/Big_Sim/"
REMOTE_PROJ_PATH="/home/seang/Dev/Git/Big_Sim/"
LOG_FILE="$CLIENT_PROJ_PATH"logs/log_remote_sign_in_error

yes_or_no() {
	while true; do
		printf "$* (y/n): "	
		read user_response
		case "$user_response" in
			[Yy]*) return 0 ;;
			[Nn]*) return 1 ;;
			*) printf "Not a valid response. Try again.\n";; 
		esac
	done
}

git_push_or_pull() {
	#FIXME: BASHVER_INFO readyonly var error	
	local -n push_or_pull=$1	
	if yes_or_no "Would you like to ${push_or_pull[0]} the project ${push_or_pull[1]} the remote?"; then 
		case "${push_or_pull[0]}" in
			"push")
				printf "\nCommiting and pushing to remote...\n"
				git add .
				read -p "Enter your commit message: " commit_message
				git commit -m "$commit_message"
				git push origin master
				;;
			"pull")
				printf "\n Pulling from remote\n"
				git pull origin master
				;;
		esac
	fi
}

run_remote() {
	command="$1"
	git_push_or_pull "$2" 
	cd $HOME/Dev/Git/Big_Sim/

	case "$command" in
		"build")
			printf "Building the entire simulation...\n" && make
			;;
		"clean")
			printf "Cleaning up object files from the simulation's main folder...\n" && \
				make clean
			;;
		"run")
			printf "Checking for executable...\n"
			if [ -x "Big_Sim" ]; then
				./Big_Sim
			else
				printf "'Big_Sim' does not exist or is not executable. Exiting...\n" && \
					return 1 
			fi
			;;
		*)
			printf "$0 expected an argument but did not receive one. Exiting...\n" && \
				return 1 
			;;
	esac
}

check_vpn() {
	printf "Checking if vpn service is running\n"

	if $(ps -e | grep -q "vpnui"); then
		printf "vpn service already running\n"
	else	
		printf "opening vpn service...\n"
		# TODO: find a vpncli	
		({ vpnui 2>/dev/null; } &) 	
	fi
}

ssh_login() {
	printf "\nAttempting to log in via ssh to $REMOTE_USER@$REMOTE_HOST...\n\n"
	ssh -T -E $LOG_FILE $REMOTE_USER@$REMOTE_HOST "$(typeset -f); "$1" "$2""
}

main() {
	check_vpn
	git_push_or_pull git_push
	printf "\nAttempting to log in via ssh to $REMOTE_USER@$REMOTE_HOST...\n\n"
	if ssh -T -E $LOG_FILE $REMOTE_USER@$REMOTE_HOST "$(typeset -f yes_or_no); \
		$(typeset -a); $(typeset -f git_push_or_pull); $(typeset -f run_remote); \
		run_remote $1 git_pull"; then
		printf "\nFinished with all tasks. Exiting...\n" 
	else	
		printf "\nAn error occurred. Exiting...\n"
	fi
}

main $1


#!/usr/bin/bash

command="$1"
git_integrate="$2"

cd $HOME/Dev/Git/Big_Sim/

if [ "$git_integrate" != "y" ] && [ "$git_integrate" != "n" ]; then
	printf "command not recognized. Exiting" && exit 1
fi

if [ "$git_integrate" == "y" ]; then
	git pull origin master
fi

case "$command" in
	"build")
		printf "Building the entire simulation...\n" && make ;;
	"clean")
		printf "Cleaning up object files from the simulation's main folder...\n" && \
			make clean ;;
	*)
		printf "$0 expected an argument but did not receive one. Exiting...\n" && \
			exit 1 ;;
esac


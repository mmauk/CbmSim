#!/usr/bin/bash

command=$1

cd $HOME/Dev/Git/Big_Sim/
case $command in
	"build")
		printf "Building the entire simulation...\n" && make ;;
	"clean")
		printf "Cleaning up object files from the simulation's main folder...\n" && \
			make clean ;;
	*)
		printf "$0 expected an argument but did not receive one. Exiting...\n" && \
			exit 1 ;;
esac


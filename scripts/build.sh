#!/usr/bin/bash 

# assume we are running this build script in Big_Sim/Big_Sim/

set -e

declare command=$1
declare OBJECT_DIR="objs/"
declare ROOT="../"
declare -a objs
declare modules=$(ls $ROOT | grep 'CBM*\|CXX*')

cd $ROOT

# drop into each module directory and compile
for module in $modules; do
   [ "$module" = "CBM_Params" ] && continue
   cd $module 
   case "$command" in 
	  "build")
		 printf "Compiling $module...\n" && make 
		 ;;
	  "clean")
		 printf "Cleaning up object files and folders from $module...\n" && make clean
		 ;;
	  *)
		 printf "$0 expected an argument but did not receive one. Exiting...\n" 
		 exit 1 
		 ;;
	esac
	cd ..
done

cd Big_Sim
case "$command" in
   "build")
	  printf "Building the entire simulation...\n" && make
	  ;;
   "clean")
	  printf "Cleaning up object files from the simulation's main folder...\n" && make clean
	  ;;
   *)
	  printf "$0 expected an argument but did not receive one. Exiting...\n"
	  exit 1
	  ;;
esac

printf "Done with all tasks. Exiting...\n"

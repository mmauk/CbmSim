#!/usr/bin/bash

printf "Cleaning all sub-directories of project files, svn files, eclipse files\n"
dirs_to_clean=$(ls | grep 'CBM*\|CXX*')
for dir in $dirs_to_clean; do
   printf "Moving into directory: $dir\n" 
   cd $dir 
   objs_to_delete=("./intout" "./lib" "./.settings" "./.svn" "./.cproject" \
	  "./linux.mk" "./.project" "./.qmake.stash" "$(find . -type f -name *.pro)")
   for obj in ${objs_to_delete[@]}; do
	  printf "deleting: $obj\n" 
	  [ -d "$obj" ] && rm -r "$obj" || [ -f "$obj" ] && rm -f "$obj"
   done
   printf "directory after cleaning (excluding .txt files):\n"
   [ "$dir" = "CBM_Params" ] && ls -a | grep -v \.txt || ls -a 
   cd ../
done

printf "done. Exiting...\n"


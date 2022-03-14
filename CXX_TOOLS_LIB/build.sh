#!/bin/bash
set -e

CBM_INC=""
CBM_LIB_PATH=""
CBM_LIB=""

INC_PATH="$CBM_INC"
LIB_PATH="$CBM_LIB_PATH"
LIBS="$CBM_LIB"

# Create the project file. Use CONFIG="debug" to debug
qmake -project -t lib INCLUDEPATH+="$INC_PATH" LIBS+="$LIB_PATH $LIBS" CONFIG="release" DESTDIR="../libs" OBJECTS_DIR="intout" -o cxx_tools.pro
# Create the makefile
qmake 
# Make the code
make

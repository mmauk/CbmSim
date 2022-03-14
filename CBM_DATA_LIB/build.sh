#!/bin/bash
set -e

CBM_INC="../CXX_TOOLS_LIB/"
CBM_LIB_PATH="-L../libs"
CBM_LIB="-lcxx_tools"

INC_PATH="$CBM_INC"
LIB_PATH="$CBM_LIB_PATH"
LIBS="$CBM_LIB"

# Create the project file. Use CONFIG="debug" to debug
qmake -project -t lib INCLUDEPATH+="$INC_PATH" LIBS+="$LIB_PATH $LIBS" CONFIG="release" DESTDIR="../libs" OBJECTS_DIR="intout" -o cbm_data.pro
# Create the makefile
qmake 
# Make the code
make

#!/bin/bash
set -e

#CUDA_INC="/opt/apps/cuda/4.2/cuda/include /usr/local/cuda/include /opt/cuda/include"
#CUDA_LIB_PATH="-L/opt/apps/cuda/4.2/cuda/lib64 -L/usr/local/cuda/lib64 -L/usr/lib64 -L/opt/cuda/lib64"
CUDA_INC="/user/local/cuda/include /usr/local/cuda/include /opt/cuda/include"
CUDA_LIB_PATH="-L/user/local/cuda/lib64 -L/usr/local/cuda/lib64 -L/usr/lib64 -L/opt/cuda/lib64"
CUDA_LIB="-lcudart"

CBM_INC="../CXX_TOOLS_LIB/ ../CBM_STATE_LIB"
CBM_LIB_PATH="-L../libs"
CBM_LIB="-lcbm_state -lcxx_tools"

INC_PATH="$CBM_INC $CUDA_INC"
LIB_PATH="$CBM_LIB_PATH $CUDA_LIB_PATH"
LIBS="$CBM_LIB $CUDA_LIB"

CUDA_SOURCES=`find . -name '*.cu'`

# Create the project file. Use CONFIG="debug" to debug
qmake -project -t lib INCLUDEPATH+="$INC_PATH" LIBS+="$LIB_PATH $LIBS" QMAKE_CXXFLAGS+="-std=c++11" CONFIG="release" DESTDIR="../libs" OBJECTS_DIR="intout" CUDA_SOURCES=$CUDA_SOURCES -o cbm_core.pro

# Add this text to the bottom of the project file to tell qmake how to compile cuda files
echo NVCCFLAGS = -O3 -arch=compute_61 -Xcompiler -fPIC >> cbm_core.pro
echo CUDA_INC = \$\$join\(INCLUDEPATH,\' -I\',\'-I\',\' \'\) >> cbm_core.pro
echo cuda.input = CUDA_SOURCES >> cbm_core.pro
echo cuda.output = \${OBJECTS_DIR}\${QMAKE_FILE_BASE}_cuda.o >> cbm_core.pro
echo cuda.commands = nvcc \$\$NVCCFLAGS \$\$CUDA_INC -c \${QMAKE_FILE_NAME} -o \${QMAKE_FILE_OUT} >> cbm_core.pro
echo cuda.dependency_type = TYPE_C >> cbm_core.pro
echo cuda.depend_command = nvcc -g -G -M \$\$NVCCFLAGS \$\$CUDA_INC \${QMAKE_FILE_NAME} >> cbm_core.pro
echo QMAKE_EXTRA_COMPILERS += cuda >> cbm_core.pro

# Create the makefile
qmake 
# Make the code
make

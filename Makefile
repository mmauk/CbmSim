##################################
# Makefile for building: cbm_sim
# Author: Sean Gallogly
# Data last modified: 12/19/2022
##################################

ROOT           := ./
BUILD_DIR      := $(ROOT)build/
DEBUG_DIR      := $(BUILD_DIR)debug/
SRC_DIR        := $(ROOT)src/
LOG_DIR        := $(ROOT)logs/
DATA_DIR       := $(ROOT)data/
DATA_IN_DIR    := $(DATA_DIR)inputs
DATA_OUT_DIR   := $(DATA_DIR)outputs
RELEASE_TARGET := $(BUILD_DIR)cbm_sim
DEBUG_TARGET   := $(DEBUG_DIR)cbm_sim

INC_DIRS  := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

ifeq (Ubuntu, $(findstring Ubuntu, $(shell uname -rv)))
	CUDA_PKG_NAME   := cuda-12.2
	CUDART_PKG_NAME := cudart-12.2
else
	CUDA_PKG_NAME   := cuda
	CUDART_PKG_NAME := cudart
endif

CUDA_INC_FLAGS := $(INC_FLAGS) $(shell pkg-config --cflags $(CUDA_PKG_NAME))
GTK_INC_FLAGS  := $(INC_FLAGS) $(shell pkg-config --cflags gtk+-3.0)

LIB_FLAGS := $(shell pkg-config --libs gtk+-3.0)
LIB_FLAGS += $(shell pkg-config --libs $(CUDART_PKG_NAME))

CUDA_SRCS := $(shell find $(SRC_DIR) -name "*.cu" | xargs -I {} basename {})
CUDA_RELEASE_OBJS := $(CUDA_SRCS:%.cu=$(BUILD_DIR)%.o)
CUDA_DEBUG_OBJS := $(CUDA_SRCS:%.cu=$(DEBUG_DIR)%.o)

NON_CUDA_SRCS := $(shell find $(SRC_DIR) -name "*.cpp" | xargs -I {} basename {})
NON_CUDA_RELEASE_OBJS := $(NON_CUDA_SRCS:%.cpp=$(BUILD_DIR)%.o)
NON_CUDA_DEBUG_OBJS := $(NON_CUDA_SRCS:%.cpp=$(DEBUG_DIR)%.o)

RELEASE_OBJS := $(CUDA_RELEASE_OBJS) $(NON_CUDA_RELEASE_OBJS)
DEBUG_OBJS   := $(CUDA_DEBUG_OBJS) $(NON_CUDA_DEBUG_OBJS)

NVCC       := nvcc
NVCC_FLAGS := -arch=native -Xcompiler -fPIC -O3

CPP             := g++-11
CPP_FLAGS       := -m64 -pipe -std=c++14 -fopenmp -O3 -fPIC
CPP_DEBUG_FLAGS := -m64 -pipe -std=c++14 -fopenmp -g -D DEBUG -fPIC 

LD             := g++-11
LD_FLAGS       := -m64 -fopenmp -O3
LD_DEBUG_FLAGS := -m64 -fopenmp -g

CHK_DIR_EXISTS   := test -d
MKDIR            := mkdir -p
RMDIR            := rmdir
RM               := rm -rf

VPATH := $(shell echo "${INC_DIRS}" | sed -e 's/ /:/g') 

first: $(LOG_DIR) $(DATA_IN_DIR) $(DATA_OUT_DIR) $(BUILD_DIR) $(RELEASE_TARGET)

debug: $(LOG_DIR) $(DATA_IN_DIR) $(DATA_OUT_DIR) $(BUILD_DIR) $(DEBUG_DIR) $(DEBUG_TARGET)

$(BUILD_DIR)%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC_FLAGS) -c $< -o $@

$(BUILD_DIR)%.o: %.cpp
	$(CPP) $(CPP_FLAGS) $(CUDA_INC_FLAGS) $(GTK_INC_FLAGS) -c $< -o $@

$(DEBUG_DIR)%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC_FLAGS) -c $< -o $@

$(DEBUG_DIR)%.o: %.cpp
	$(CPP) $(CPP_DEBUG_FLAGS) $(CUDA_INC_FLAGS) $(GTK_INC_FLAGS) -c $< -o $@

$(RELEASE_TARGET): $(RELEASE_OBJS)
	$(LD) $(LD_FLAGS) $^ -o $@ $(LIB_FLAGS) 

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(LD) $(LD_DEBUG_FLAGS) $^ -o $@ $(LIB_FLAGS) 

$(LOG_DIR):
	@$(CHK_DIR_EXISTS) $(LOG_DIR) || $(MKDIR) $(LOG_DIR)

$(DATA_IN_DIR):
	@$(CHK_DIR_EXISTS) $(DATA_IN_DIR) || $(MKDIR) $(DATA_IN_DIR)

$(DATA_OUT_DIR):
	@$(CHK_DIR_EXISTS) $(DATA_OUT_DIR) || $(MKDIR) $(DATA_OUT_DIR)

$(BUILD_DIR):
	@$(CHK_DIR_EXISTS) $(BUILD_DIR) || $(MKDIR) $(BUILD_DIR)

$(DEBUG_DIR):
	@$(CHK_DIR_EXISTS) $(DEBUG_DIR) || $(MKDIR) $(DEBUG_DIR)

.PHONY: clean
clean:
	$(RM) $(BUILD_DIR)*.o
	$(RM) $(RELEASE_TARGET)


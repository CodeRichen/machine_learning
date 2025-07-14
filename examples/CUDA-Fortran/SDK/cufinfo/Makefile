# SPDX-FileCopyrightText: Copyright (c) 2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FC = nvfortran
OBJ = o
EXE = out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ = obj
EXE = exe
endif

FCFLAGS ?= -fast

# Uncomment the CUDAFLAGS line if you have hardware requirements that
# require a specific compute capability
# CUDAFLAGS ?= -cuda

all: build run verify

build: cufinfo.cuf
	$(FC) $(FCFLAGS) $(CUDAFLAGS) -o cufinfo.$(EXE) $<

run: cufinfo.$(EXE)
	$(RUN) ./cufinfo.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf cufinfo.$(EXE) *.$(OBJ) *.dwf *.pdb prof

# SPDX-FileCopyrightText: Copyright (c) 2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

NTHREADS = 1
CC = nvc
FC = nvfortran
OBJ = o
EXE = out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ = obj
EXE = exe
endif

CCFLAGS ?= -fast -Mpreprocess
FCFLAGS ?= -fast -Mconcur -Minline=saxpy,sscal -Minfo -Mpreprocess

all: build run verify

build: linpkrd.f second.c
	@echo ------------------------------------ building linpkrd.$(EXE)
	$(CC) $(CCFLAGS) -c -o second.$(OBJ) second.c
	$(FC) $(FCFLAGS) -o linpkrd.$(EXE) linpkrd.f second.$(OBJ)

run: linpkrd.$(EXE)
	@echo '------------- executing linpkrd.$(EXE) with $(NTHREADS) threads'
	export OMP_NUM_THREADS=$(NTHREADS); ./linpkrd.$(EXE)

verify: run

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(OBJ) linpkrd.$(EXE) *.dwf *.pdb prof

# SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FC = nvfortran
NTHREADS = 1
OBJ = o
EXE = out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ = obj
EXE = exe
endif

FCFLAGS ?= -Mconcur -fast -Minfo

all: build run verify

build: matmul.f90
	@echo ------------------------------------ building matmul.$(EXE)
	$(FC) $(FCFLAGS) -o matmul.$(EXE) matmul.f90

run: matmul.$(EXE)
	@echo '------------- executing matmul.$(EXE) with $(NTHREADS) threads'
	export OMP_NUM_THREADS=$(NTHREADS); ./matmul.$(EXE)

verify: run

clean:
	@echo 'Cleaning up...'
	@rm -rf matmul.$(EXE) *.dwf *.pdb prof

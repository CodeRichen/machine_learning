# SPDX-FileCopyrightText: Copyright (c) 2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

NTHREADS = 2
OBJ = o
EXE = out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ = obj
EXE = exe
endif

FCFLAGS ?= -fast -mp -Minfo

all: build run verify

build: matmul.F
	@echo ------------------------------------ building matmul.$(EXE)
	nvfortran $(FCFLAGS) -DPGF90 -DSYSCLK -o matmul.$(EXE) matmul.F

run: matmul.$(EXE)
	@echo '------------- executing matmul.$(EXE) with $(NTHREADS) threads'
	OMP_NUM_THREADS=$(NTHREADS) $(RUN) ./matmul.$(EXE)

verify: run

clean:
	@echo 'Cleaning up...'
	@rm -rf matmul.$(EXE) *.dwf *.pdb prof

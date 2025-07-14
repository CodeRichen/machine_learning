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
OBJ	= o
EXE	= out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ	= obj
EXE = exe
endif

FCFLAGS ?= -fast -cuda -cudalib=cublas

all: build run verify

build: cublasTestSgemm.F90
	$(FC) $(FCFLAGS) -o sgemm_gpu.$(EXE) $<
	$(FC) $(FCFLAGS) -o sgemm_host.$(EXE) $< -lblas

run: sgemm_gpu.$(EXE) sgemm_host.$(EXE)
	@echo '----------------------------- GPU run ----------------'
	$(RUN) ./sgemm_gpu.$(EXE)
	@echo '----------------------------- CPU run ----------------'
	$(RUN) ./sgemm_host.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf sgemm_gpu.$(EXE) cublasTestSgemm.$(OBJ) sgemm_host.$(EXE) *.dwf *.pdb prof

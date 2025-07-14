# SPDX-FileCopyrightText: Copyright (c) 2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

OBJ = o
EXE = out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ = obj
   EXE = exe
endif

FCFLAGS ?= -Minfo

all: build run verify

build: adj3.f90
	@echo ------------------------------------ building adj3.$(EXE)
	nvfortran $(FCFLAGS) -o adj3.$(EXE) adj3.f90

run: adj3.$(EXE)
	@echo '------------- executing adj3.$(EXE) '
	./adj3.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.mod adj3.$(EXE) *.dwf *.pdb prof

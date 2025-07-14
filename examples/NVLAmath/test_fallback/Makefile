# SPDX-FileCopyrightText: Copyright (c) 2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

TEST = lu_fallback
FC = nvfortran
EXE = exe

FCFLAGS ?= -cuda -gpu=nvlamath -cudalib=nvlamath -O2

all: build run verify

build: $(TEST).F90
	$(FC) $(FCFLAGS) -o $(TEST).$(EXE) $<

run: $(TEST).$(EXE)
	export NV_LAMATH_FALLBACK=1; \
	$(RUN) ./$(TEST).$(EXE)
	@echo 'This is a negative example. It is supposed to print error messsages and terminate the application.'
	$(RUN) ./$(TEST).$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof

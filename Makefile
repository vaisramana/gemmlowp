UNITTESTS_COMMON=benchmark.cc test.cc test_allocator.cc test_blocking_counter.cc test_fixedpoint.cc test_math_helpers.cc
UNITTESTS_X86=$(UNITTESTS_COMMON)

UNITTESTS_ARM=$(UNITTESTS_COMMON)

UNITTESTS_X86_BIN=$(addprefix ./test/, $(addsuffix .x86, $(basename $(UNITTESTS_X86))))
UNITTESTS_ARM_BIN=$(addprefix ./test/, $(addsuffix .arm, $(basename $(UNITTESTS_ARM))))

#UNITTESTS_BIN=$(UNITTESTS_X86_BIN)
UNITTESTS_BIN=$(UNITTESTS_ARM_BIN)

VPATH=./test ./public

space :=
space +=
join-with = $(subst $(space),$1,$(strip $2))

.PHONY: compile clean unittest

#CC_X86=clang++
TOOLCHAIN_ROOT=/home/shuo/shuo/bin/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin
CC = $(TOOLCHAIN_ROOT)/aarch64-linux-gnu-g++

CC_X86=c++
CFLAGS_X86=-march=native -O3 -pthread
CFLAGS_ARM=-O3 -pthread

compile: $(UNITTESTS_BIN)

clean:
	rm -f $(UNITTESTS_BIN)

unittest: $(UNITTESTS_BIN)
	$(call join-with, && ,$(addprefix ./, $^))

#%.x86: %.cc ./eight_bit_int_gemm/eight_bit_int_gemm.cc ./test/test_data.cc
#	$(CC_X86) $(CFLAGS_X86) -std=c++11 -g -O3 -o $@ $^


%.arm: %.cc ./eight_bit_int_gemm/eight_bit_int_gemm.cc ./test/test_data.cc
	$(CC) $(CFLAGS_ARM) -std=c++11 -g -O3 -o $@ $^

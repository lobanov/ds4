# Research-only DSpark drafter GGUF converter (copy of gguf-tools tooling).
# See README.md in this directory for provenance.
CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -std=c11
CPPFLAGS ?= -D_GNU_SOURCE
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

.PHONY: all clean

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(NATIVE_CPU_FLAG) -o $@ deepseek4-quantize.c quants.c -lm -pthread

clean:
	rm -f deepseek4-quantize

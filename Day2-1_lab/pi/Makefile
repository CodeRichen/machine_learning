# Makefile for mpi-monte-carlo.c

# Compiler
CC=mpicc

# Compiler flags for Optimization
CFLAGS=-Wall -funroll-loops -O3 -flto -march=native

# Target executable name
TARGET=monte_carlo_mpi

all: $(TARGET)

$(TARGET): monte_carlo.c
	$(CC) $(CFLAGS) -o $(TARGET) monte_carlo.c

clean:
	rm -f $(TARGET)
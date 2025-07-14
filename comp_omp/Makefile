CXX = g++
CXXFLAGS = -O3

.PHONY: all
all: n-body

n-body: n-body.cpp
	$(CXX) $(CXXFLAGS) -I/work/HPC_SCAMP/2024/files/comp_omp/include -o n-body n-body.cpp -fopenmp

.PHONY: clean
clean:
	rm -f n-body
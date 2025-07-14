CXX = g++
CXXFLAGS = -O3

.PHONY: all
all: bar_chart

bar_chart: bar_chart.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o bar_chart bar_chart.cpp -I/opt/ohpc/Taiwania3/pkg/local/Python/3.12.2/include/python3.12 -L/opt/ohpc/Taiwania3/pkg/local/Python/3.12.2/lib -lpython3.12

.PHONY: clean
clean:
	rm -f bar_chart

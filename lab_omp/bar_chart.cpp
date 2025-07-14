#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Error: No input file provided." << std::endl;
        return 1;
    }
    char *inputfile_name = argv[1];

    if (argc < 3)
    {
        std::cerr << "Error: No output file provided." << std::endl;
        return 1;
    }
    char *outputfile_name = argv[2];

    std::ifstream infile(inputfile_name);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file: " << inputfile_name << std::endl;
        return 1;
    }

    std::cout << "Reading input file..." << std::endl;

    std::string line;
    std::getline(infile, line);
    int n = std::stoi(line);

    std::vector<std::vector<int>> array(n, std::vector<int>(5));
    for (int i = 0; i < n; i++)
    {
        if (!std::getline(infile, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            return 1;
        }
        std::istringstream ss(line);
        for (int j = 0; j < 5; j++)
        {
            if (!(ss >> array[i][j])) {
                std::cerr << "Error: Invalid data format" << std::endl;
                return 1;
            }
        }
    }
    infile.close();

    std::vector<std::vector<int>> grade_span(5, std::vector<int>(5, 0));

    std::cout << "Finish reading input file. Total " << n << " records." << std::endl;
    std::cout << "Start sorting..." << std::endl;
    auto start = std::chrono::steady_clock::now();

    // 使用OpenMP并行处理，使用collapse(2)合并两个循环
    #pragma omp parallel
    {
        // std::vector<std::vector<int>> local_grade_span(5, std::vector<int>(5, 0)); 
int local_grade_span[5][5] = {0};
        // #pragma omp for collapse(2) schedule(dynamic)
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < 5; k++)
            {
                int grade = array[i][k];
                if (grade <= 100 && grade >= 90)
                {
                    local_grade_span[k][0]++;
                }
                else if (grade < 90 && grade >= 80)
                {
                    local_grade_span[k][1]++;
                }
                else if (grade < 80 && grade >= 70)
                {
                    local_grade_span[k][2]++;
                }
                else if (grade < 70 && grade >= 60)
                {
                    local_grade_span[k][3]++;
                }
                else if (grade < 60 && grade >= 0)
                {
                    local_grade_span[k][4]++;
                }
                else
                {
                    std::cerr << "Error: Invalid grade " << grade << std::endl;
                }
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < 5; k++)
            {
                for (int l = 0; l < 5; l++)
                {
                    grade_span[k][l] += local_grade_span[k][l];
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Sorting complete." << std::endl;
    std::cout << "Sorting Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::cout << "Start plotting..." << std::endl;

    for (int i = 0; i < 5; i++)
    {
        std::vector<double> x = {0 + 0.8 * i, 5 + 0.8 * i, 10 + 0.8 * i, 15 + 0.8 * i, 20 + 0.8 * i};
        std::vector<double> h = {(double)grade_span[i][0], (double)grade_span[i][1], (double)grade_span[i][2], (double)grade_span[i][3], (double)grade_span[i][4]};
        plt::bar(x, h);
    }
    std::vector<double> x_positions = {1.6, 6.6, 11.6, 16.6, 21.6};
    std::vector<std::string> x_labels = {"100-90", "90-80", "80-70", "70-60", "below 60"};
    plt::xticks(x_positions, x_labels);
    plt::save(outputfile_name);
    std::cout << "Plotting complete." << std::endl;
    std::cout << "Saved as " << outputfile_name << std::endl;

    return 0;
}
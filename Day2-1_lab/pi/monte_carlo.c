#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef N
#define N 1000000000 // 1e9
#endif

#ifdef M_PIl
#define PI M_PIl
#else
#define PI 3.141592653589793238462643383279502884L
#endif

int main(int argc, char* argv[])
{
    // Ensure only one additional argument is provided
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_tosses>\n", argv[0]);
        return 1;
    }

    const long long int n = strtoull(argv[1], NULL, 10);
    if (n <= 0) {
        fprintf(stderr, "Number of tosses must be a positive integer.\n");
        return 1;
    }

    clock_t tik = clock();
    srand(time(NULL)); // Seed random number generator

    long long int number_in_circle = 0;

    for (long long int toss = 0; toss < n; ++toss) {
        double x = (double)rand() / (RAND_MAX + 1.0);
        double y = (double)rand() / (RAND_MAX + 1.0);
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) {
            number_in_circle++;
        }
    }

    double pi_estimate = 4 * (double)number_in_circle / n;

    if (isnan(pi_estimate)) {
        fprintf(stderr, "pi_estimate calculation resulted in NaN.\n");
        return 1;
    }

    double abs_err = fabsl(pi_estimate - PI);

    if (isnan(abs_err)) {
        fprintf(stderr, "Absolute error calculation resulted in NaN.\n");
        return 1;
    }

    printf("pi_estimate\t= %.15lf\nAbs err\t= %.15lf\n", pi_estimate, abs_err);

    clock_t tok = clock();
    double time_spent = (double)(tok - tik) / CLOCKS_PER_SEC;
    fprintf(stderr, "Wall time: %f seconds\n", time_spent);

    return 0;
}

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 


    srand(time(NULL) + world_rank);


    long long total_tests_per_process = strtoll(argv[1], NULL, 10) / world_size;
    long long local_circle_count = 0;

    for (long long i = 0; i < total_tests_per_process; ++i) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_circle_count++;
        }
    }


    long long global_circle_count;
    MPI_Reduce(&local_circle_count, &global_circle_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);


    if (world_rank == 0) {
        double pi_estimate = 4.0 * global_circle_count / (total_tests_per_process * world_size);
        cout << "Estimated Pi = " << pi_estimate << endl;
    }

    MPI_Finalize();

    return 0;
}

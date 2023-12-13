// Programme to test the multi-processor speed up gain, if any on the host system
// compile the code in the following way:
    // mpic++ -o multiCore_speed_test multiCore_speed_test.cpp
// run using : mpirun -np 16 ./multiCore_speed_test


#include <iostream>
#include <cmath>
#include <mpi.h>

double computeSquareRoot(int start, int end) {
    double result = 0;
    for (int i = start; i <= end; ++i) {
        result += std::sqrt(i);
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long int limit = 1000000000000;

    if (rank == 0) {
        std::cout << "Number of processors: " << size << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize all processes before starting the computation

    double localResult = 0;
    int localStart = 1 + rank * (limit / size);
    int localEnd = (rank + 1) * (limit / size) - 1;

    double startTime = MPI_Wtime();
    localResult = computeSquareRoot(localStart, localEnd);
    double endTime = MPI_Wtime();

    double totalResult;
    MPI_Reduce(&localResult, &totalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsedTime = endTime - startTime;
    double maxElapsedTime;
    MPI_Reduce(&elapsedTime, &maxElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total execution time on " << size << " processors: " << maxElapsedTime << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}

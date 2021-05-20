#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

typedef unsigned char byte;

int main(int argc, char **argv)
{
    unsigned n = 100000;
    double sum = 0.;
    int rank, size; // MPI

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
    if (rank == 0)
        scanf(" %u", &n);
        */
    const double starttime = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (unsigned i = rank; i < n; i += size)
    {
        double x = (i + 0.5) / n;
        sum += 4.0 / (1.0 + x * x);
    }
    double total;
    MPI_Reduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    total /= n;
    const double endtime = MPI_Wtime();
    MPI_Finalize();
    // printf("%d\n", rank);

    if (rank == 0)
        printf("pi: %.12lf\nCost time: %.12lf s\n", total, endtime - starttime);
    return 0;
}
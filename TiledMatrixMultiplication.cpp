// Headers
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define tile_size 64
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define SEED 0

/**
 * A function to multiply matrices serially.
 * @param a : Matrix A (nXn)
 * @param b : Matrix B (nXn)
 * @param c : Matrix C to hold result (nXn)
 * @param n : Dimension of Square matrix
 * @return None
*/
void matrix_multiplication(double **a, double **b, double **d, int n)
{
    double sum;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            d[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                d[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}


/**
 * A function to create matrix with random entries.
 * @param n : Dimension of Square matrix
 * @return Matrix 
*/
double **create_matrix(int n)
{
    int i, j;
    double **a;
    a = (double **)malloc(sizeof(double *) * n);
    for (i = 0; i < n; i++)
    {
        a[i] = (double *)malloc(sizeof(double) * n);
    }
    srand(SEED);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i][j] = rand() % 10;
        }
    }
    return a;
}


/**
 * A function to free dynamically allocated matrix.
 * @param a : Matrix A (nXn)
 * @param n : Dimension of Square matrix
 * @return None
*/
void free_matrix(double **a, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        free(a[i]);
    }
    free(a);
}


// main method
int main(int argc, char *argv[])
{

    int n;
    if (argc != 2)
    {
        printf("IMproper Arguments!!");
        exit(0);
    }

    n = atoi(argv[1]);

    struct timeval TimeValue_Start;
    struct timezone TimeZone_Start;

    struct timeval TimeValue_Final;
    struct timezone TimeZone_Final;

    long time_start, time_end;
    double time_overhead;
    int i, j, k, ii, jj, kk, sum;

    double **A, **B, **C, **D;
    double t1, t2, t3, t4;

    // Create matrix A and B
    A = create_matrix(n);
    B = create_matrix(n);

    C = (double **)malloc(sizeof(double *) * n);
    for (i = 0; i < n; i++)
    {
        C[i] = (double *)malloc(sizeof(double) * n);
    }
    D = (double **)malloc(sizeof(double *) * n);
    for (i = 0; i < n; i++)
    {
        D[i] = (double *)malloc(sizeof(double) * n);
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i][j] = 0.0;
            D[i][j] = 0.0;
        }
    }

    gettimeofday(&TimeValue_Start, &TimeZone_Start);
    matrix_multiplication(A, B, D, n);
    gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead = (time_end - time_start) / 1000000.0;

    printf("\n\t........................................................................\n");
    printf(" \t          Matrix Multiplication (Serial VS Tiled)                          ");
    printf("\n\t........................................................................\n");

    printf("\n\n\t\t Matrix into Matrix Multiplication (Serial) ......Done \n");
    printf("\n\t\t Time in Seconds (T)        : %lf Seconds \n", time_overhead);
    printf("\n\t\t ( T represents the Time taken for computation )");

    // Parallel Implementation
    gettimeofday(&TimeValue_Start, &TimeZone_Start);
    #pragma omp parallel for private(jj, kk, i, j, k, sum)
        for (ii = 0; ii < n; ii += tile_size)
        {
            for (jj = 0; jj < n; jj += tile_size)
            {
                for (kk = 0; kk < n; kk += tile_size)
                {
                    for (i = ii; i < MIN(ii + tile_size, n); i++)
                    {
                        for (j = jj; j < MIN(jj + tile_size, n); j++)
                        {
                            sum = 0;
                            for (k = kk; k < MIN(kk + tile_size, n); k++)
                            {
                                sum += A[i][k] * B[k][j];
                            }
                            C[i][j] += sum;
                        }
                    }
                }
            }
        }
    gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead = (time_end - time_start) / 1000000.0;

    printf("\n\n\t\t Matrix into Matrix Multiplication (Tiled) ......Done \n");
    printf("\n\t\t Time in Seconds (T)        : %lf Seconds \n", time_overhead);
    printf("\n\t\t ( T represents the Time taken for computation )");
    printf("\n\t\t..........................................................................\n");

    // Verify the results
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            if (C[i][j] != D[i][j])
            {
                printf("Failed");
                exit(-1);
            }
        }

    printf("Done Checking : Both Computations are same!! ");
    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);
    free_matrix(D, n);

    return 0;
}
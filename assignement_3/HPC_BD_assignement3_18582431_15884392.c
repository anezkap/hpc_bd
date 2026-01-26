/******************************************************************************
* FILE: HPC_BD_assignement3_18582431_15884392.c
* DESCRIPTION: 
*   This our implementation
* AUTHOR: Balthazar Dupuy d'Angeac 18582431 & Anezka Potesilova 15884392
* LAST REVISED: 22/01/26
******************************************************************************/

#include "mpi.h"
// Includes the MPI header file, which provides the necessary functions and definitions for MPI programs.
#include <stdio.h>
// Includes the standard I/O library for functions like printf.
#include <stdlib.h>
// Includes the standard integer types library.
#include <stdint.h>
// Includes the standard library for general utilities like memory management.
#include <string.h>
// Includes the string library for functions like memset.
#include <unistd.h>

#define  MASTER		0
#define MATRIX_SIZE 10  // TODO: change to 3000
#define niter 2    // TODO: change to 5000


int *get_nbours(int pid, int m, int n);
void exchange_ghost_cells(int *matrix, int num_rows, int num_cols, int *neighbours);
int calculate_number_live_cells(int *matrix, int num_rows, int num_cols);

int main (int argc, char *argv[])
// Main function for the program. `argc` and `argv` handle command-line arguments.
{
    // Variables to store the number of ranks, rank, destination, tag, source, m (sub-matrix with contour input of update), newm (sub-matrix without contour output of update) and message count.
    int i, j, proc, iter, nrank, rank;

    // Structure to hold information about the status of an MPI operation.
    MPI_Status Stat;

    // Initializes the MPI environment. Must be called before any other MPI functions.
    MPI_Init(&argc,&argv);

    // Determines the total number of processes (tasks) in the MPI communicator `MPI_COMM_WORLD`.
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    
    // Determines the rank (ID) of the calling process within the communicator.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // find/update m, n such that processes are given "the most square sub-matrix possible" to work with
    // guaranteeing to minimize communication overhead
    int dims[2] = {0, 0};
    MPI_Dims_create(nrank, 2, dims);

    int m = dims[0];   // rows
    int n = dims[1];   // cols

    if (rank == MASTER) {
        printf("Rank %d: matrix partitioning in %d rows and %d cols\n", rank, m, n);
    }

    // initialize "subM": the local sub-matrix that each process will update
    int subM_rows = MATRIX_SIZE/m; // number of rows in sub-matrix without ghost cells
    int subM_cols = MATRIX_SIZE/n; // number of columns in sub-matrix without ghost cells
    int subM[subM_rows + 2][subM_cols + 2] = {};

    // Logic for task with Master rank.
    if (rank == MASTER) {
        #define PATTERN_HEIGHT 6
        #define PATTERN_WIDTH 6

        uint8_t pattern[PATTERN_HEIGHT][PATTERN_WIDTH] = {
            {1, 1, 1, 1, 1, 1},
            {1, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 1},
            {1, 1, 1, 1, 1, 1},
        };

        int i, j, M[MATRIX_SIZE][MATRIX_SIZE] = {0}; // 2D array

        // initialize the pattern of our game of life implementation
        // implement the pattern in our initial grid
        for (i = 0; i < PATTERN_HEIGHT; i++) {
            for (j = 0; j < PATTERN_WIDTH; j++) {
                M[(MATRIX_SIZE - PATTERN_HEIGHT)/2 + i][(MATRIX_SIZE - PATTERN_WIDTH)/2 + j] = pattern[i][j];
            }
        }

        //Send portions of M to every other processes
        for (proc = nrank - 1; proc >= 0; proc--)
        {
            
            for (i = 0; i < subM_rows; i++)
            {
                for (j = 0; j < subM_cols; j++)
                {
                    subM[i+1][j+1] = M[(proc/n)*subM_rows + i][(proc%n)*subM_cols + j];
                }
            }

            //Send sub-matrix to each process
            if (proc != MASTER)
            {
                MPI_Send(&subM[0][0],
                        (subM_rows + 2) * (subM_cols + 2),
                        MPI_INT,
                        proc,
                        0,
                        MPI_COMM_WORLD);
            }
        }
    }

    // Logic for task with non-Master ranks.
    else if (rank != MASTER) {
        // Receive sub-matrix from master process
        MPI_Recv(&subM[0][0],
         (subM_rows + 2) * (subM_cols + 2),
         MPI_INT,
         MASTER,
         0,
         MPI_COMM_WORLD,
         &Stat);
    }

    int *neighbours = get_nbours(rank, m, n);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double simulation_start = MPI_Wtime();

    // Run the simulation for niter iterations
    for (iter = 0; iter < niter; iter++)
    {
        exchange_ghost_cells(&subM[0][0], subM_rows+2, subM_cols+2, neighbours);
        
        //update sub-matrix
        int temp[subM_rows+2][subM_cols+2]= {};
        int cols = subM_cols+2;
        for (i = 0; i < subM_rows; i++)
        {
            for (j = 0; j < subM_cols; j++)
            {
                int nsum = 0;
                nsum = subM[i][j] + subM[i][j+1] + subM[i][j+2] + subM[i+2][j] +
                    subM[i+2][j+1] + subM[i+2][j+2] + subM[i+1][j] + subM[i+1][j+2];
                if (nsum == 3 || (subM[i+1][j+1] == 1 && nsum == 2))
                {
                    temp[i+1][j+1] = 1;
                }
            }
        }
        memcpy(subM, temp, sizeof(int) * (subM_rows + 2) * (subM_cols + 2));

        if ((iter+1) % 10 == 0) {       // TODO: change to 10 or whatever
            int local_live = calculate_number_live_cells(&subM[0][0], subM_rows+2, subM_cols+2);
            int global_live = 0;
        
            MPI_Reduce(&local_live, &global_live, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
            if (rank == 0) {
                printf("[Iteration %d] [generation %d] Total live cells = %d\n", iter, iter+1, global_live);
            }
        }

    }

    // Stop timing and print result
    double simulation_end = MPI_Wtime();
    double local_time = simulation_end - simulation_start;
    double max_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        printf("Script finalised after %d iterations after %f seconds\n", niter, max_time);
    }

    // Gather the data into the master process
    if (rank == MASTER) {
        int M[MATRIX_SIZE][MATRIX_SIZE] = {0};

        // Master process gathers sub-matrices from all processes
        for (proc = 0; proc < nrank; proc++)
        {
            if (proc == MASTER) {
                // Copy its own sub-matrix into the correct position in M
                for (i = 0; i < subM_rows; i++)
                {
                    for (j = 0; j < subM_cols; j++)
                    {           
                        M[i][j] = subM[i+1][j+1];
                    }
                }
            } else {
                // Receive sub-matrix from other processes
                MPI_Recv(&subM[0][0],
                         (subM_rows + 2) * (subM_cols + 2),
                         MPI_INT,
                         proc,
                         3,
                         MPI_COMM_WORLD,
                         &Stat);
                
                // Copy received sub-matrix into the correct position in M
                for (i = 0; i < subM_rows; i++)
                {
                    for (j = 0; j < subM_cols; j++)
                    {           
                        M[(proc/n)*subM_rows + i][(proc%n)*subM_cols + j] = subM[i+1][j+1];
                    }
                }
            }
        }

        // Print the final matrix M
        printf("Final Matrix M after %d iterations:\n", niter);
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                printf("%d ", M[i][j]);
            }
            printf("\n");
        }
    }

    else {
        // Non-master processes send their sub-matrix back to the master process
        MPI_Send(
            &subM[0][0],
            (subM_rows + 2) * (subM_cols + 2),
            MPI_INT,
            MASTER,
            3,
            MPI_COMM_WORLD
        );
    }

    // Terminates the MPI environment. Must be the last MPI call in the program.
    free(neighbours);
    MPI_Finalize();
}
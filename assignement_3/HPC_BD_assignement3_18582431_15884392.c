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
// Includes the standard library for general utilities like memory management.
#include <string.h>
#define  MASTER		0
#define MATRIX_SIZE 6

int main (int argc, char *argv[])
// Main function for the program. `argc` and `argv` handle command-line arguments.
{
    // Variables to store the number of ranks, rank, destination, tag, source, m (sub-matrix with contour input of update), newm (sub-matrix without contour output of update) and message count.
    int i, j, proc, iter, nrank, rank, dest, tag, source, newm, comm, niter=10;

    // Variables for message passing: `inmsg` is the received message, and `outmsg` is the message to send.
    char inmsg, outmsg='x', outmsg_1MB[1024*2056], inmsg_1MB[1024*2056];
       // Initialize the 1MB message (or 1 character for initial test)
    memset(outmsg_1MB, 'x', sizeof(outmsg_1MB));

    // Structure to hold information about the status of an MPI operation.
    MPI_Status Stat;

    // Variable to hold the request handle for non-blocking communication.
    MPI_Request request;

    // Initializes the MPI environment. Must be called before any other MPI functions.
    MPI_Init(&argc,&argv);

    // Determines the total number of processes (tasks) in the MPI communicator `MPI_COMM_WORLD`.
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    
    // Determines the rank (ID) of the calling process within the communicator.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // find/update m, n such that processes are given "the most square sub-matrix possible" to work with
    // guaranteeing to minimize communication overhead
    int n = nrank;
    int m = 1;
    while (m < n) {
        m *= 2;
        n /= 2;
    }

    // initialize "subM": the local sub-matrix that each process will update
    int subM[MATRIX_SIZE/m + 2][MATRIX_SIZE/n + 2] = {};

    // Logic for task with Master rank.
    if (rank == MASTER) {
        #define PATTERN_HEIGHT 5
        #define PATTERN_WIDTH 5

        uint8_t pattern[PATTERN_HEIGHT][PATTERN_WIDTH] = {
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4}
        };

        int niter, i, j, M[MATRIX_SIZE][MATRIX_SIZE] = {0}; // 2D array
        // initialize the pattern of our game of life implementation
        // implement the pattern in our initial grid
        for (i = 0; i < PATTERN_HEIGHT; i++) {
            for (j = 0; j < PATTERN_WIDTH; j++) {
                M[(MATRIX_SIZE - PATTERN_HEIGHT)/2 + i][(MATRIX_SIZE - PATTERN_WIDTH)/2 + j] = pattern[i][j];
            }
        }

        // print initial matrix
        printf("Initial Matrix M:\n");
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                printf("%d ", M[i][j]);
            }
            printf("\n");
        }


        //Send portions of M to every other processes
        for (proc = nrank - 1; proc >= 0; proc--)
        {
            
            for (i = 0; i < MATRIX_SIZE/m; i++)
            {
                for (j = 0; j < MATRIX_SIZE/n; j++)
                {
                    subM[i+1][j+1] = M[(proc/n)*MATRIX_SIZE/m + i][(proc%n)*MATRIX_SIZE/n + j];
                }
            }

            //Send sub-matrix to each process
            if (proc != MASTER)
            {
                MPI_Isend(&subM[0][0],
                        (MATRIX_SIZE/m + 2) * (MATRIX_SIZE/n + 2),
                        MPI_INT,
                        proc,
                        0,
                        MPI_COMM_WORLD,
                        &request);
            }
        }
    }
    // Logic for task with non-Master ranks.
    else if (rank != MASTER) {
        // Receive sub-matrix from master process
        MPI_Recv(&subM[0][0],
         (MATRIX_SIZE/m + 2) * (MATRIX_SIZE/n + 2),
         MPI_INT,
         MASTER,
         0,
         MPI_COMM_WORLD,
         &Stat);
    }

    MPI_Barrier(MPI_COMM_WORLD); // synchronize all processes before printing
    // print the chunk assigned to each process
    
    for (int p = 0; p < nrank; p++) {
    if (rank == p) {
        // your existing printf block
        printf("\n I, proc %d, have received chunk: \n", rank);
        for (i = 0; i < MATRIX_SIZE/m+2; i++)
        {
            for (j = 0; j < MATRIX_SIZE/n+2; j++)
            {
                printf("%d ", subM[i][j]);
            }
            printf("\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // synchronize before next process prints
}
    
    
    // Terminates the MPI environment. Must be the last MPI call in the program.
    MPI_Finalize();
}
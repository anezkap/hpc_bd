/******************************************************************************
* FILE: HPC_BD_assignement3_18582431_15884392.c
* DESCRIPTION: 
*   This is our implementation of Conway's Game of Life over a 3000x3000 grid distributed
*   across multiple processes using MPI for parallel processing.
* AUTHOR: Balthazar Dupuy d'Angeac 18582431 & Anezka Potesilova 15884392
* LAST REVISED: 28/01/26
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

#define MASTER	0
#define MATRIX_SIZE 10  // TODO: change to 3000
#define niter 0    // TODO: change to 5000
#define min(a,b) ((a) < (b) ? (a) : (b)) // Macro to find the minimum of two values.

int *get_nbours(int pid, int m, int n);
void exchange_ghost_cells(int *matrix, int num_rows, int num_cols, int *neighbours);
int calculate_number_live_cells(int *matrix, int num_rows, int num_cols);

int main (int argc, char *argv[])
// Main function for the program. `argc` and `argv` handle command-line arguments.
{
    // initialize loop variables
    int i, j, proc;

    // Variables to store the number of iterations, number of ranks and current rank.
    int iter, nrank, rank;

    // Structure to hold information about the status of an MPI operation.
    MPI_Status Stat;

    // Initializes the MPI environment.
    MPI_Init(&argc,&argv);

    // Determines "nrank": the total number of processes (tasks) in the MPI communicator `MPI_COMM_WORLD`.
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    
    // Determines the rank (ID) of the calling process within the communicator.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // find/update m, n such that m + n is minimized
    // guaranteeing to minimize communication overhead
    int m = 1, n = nrank;
    while (m < n)
    {
        m *=2;
        n /=2;
    };

    // if (rank == MASTER) {
    //     printf(" %d rows and %d cols\n", m, n);
    // }

    int x = MATRIX_SIZE% m;  // number of extra rows to be still allocated
    int y = MATRIX_SIZE% n;  // number of extra columns to be still allocated

    // if (rank == MASTER) {
    //     printf(" %d rows and %d cols\n", m, n);
    // }

    int subM_rows = MATRIX_SIZE/m; // starting number of rows in sub-matrix without ghost cells
    int subM_cols = MATRIX_SIZE/n; // starting number of columns in sub-matrix without ghost cells

    // values taking 1 if extra row/column needed and 0 otherwise
    int x_sup = (rank / n < x);
    int y_sup = (rank % n < y);
    // printf("Rank %d: sub-matrix size %d x %d with extra row: %d and extra column: %d\n", rank, subM_rows, subM_cols, x_sup, y_sup);

    // initialize "subM": the local sub-matrix that each process will update
    int subM[subM_rows + x_sup+ 2][subM_cols + y_sup + 2];
    memset(subM, 0, sizeof(subM));

    if (rank == MASTER) {
        printf("Rank %d: matrix partitioning in %d rows and %d cols\n", rank, m, n);
    }

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
        {1, 1, 1, 1, 1, 1}
    };
        // initialize the main matrix M
        int i, j, M[MATRIX_SIZE][MATRIX_SIZE] = {0}; // 2D array

        // implement the pattern in our initial grid
        for (i = 0; i < PATTERN_HEIGHT; i++) {
            for (j = 0; j < PATTERN_WIDTH; j++) {
                M[(MATRIX_SIZE - PATTERN_HEIGHT)/2 + i][(MATRIX_SIZE - PATTERN_WIDTH)/2 + j] = pattern[i][j];
            }
        }


        if (rank == MASTER) {
            printf("Original Matrix M after %d iterations:\n", niter);
            for (i = 0; i < MATRIX_SIZE; i++) {
                for (j = 0; j < MATRIX_SIZE; j++) {
                    printf("%d ", M[i][j]);
                }
                printf("\n");
            }
        }

        //Send portions of M to every other processes (done in reverse order so the last one remains in master)
        //we allocate extra rows/columns to the first x/y processes respectively
        for (proc = nrank - 1; proc >= 0; proc--)
        {
            int x_sup = (proc / n < x);
            int y_sup = (proc % n < y);

            int subM[subM_rows + x_sup + 2][subM_cols + y_sup + 2] = {};
            for (i = 0; i < subM_rows + x_sup; i++)
                {
                    for (j = 0; j < subM_cols + y_sup; j++)
                    {
                        // copy portion of M into subM for this process
                        subM[i+1][j+1] = M[(proc/n)*subM_rows + i + min(proc / n, x)][(proc%n)*subM_cols + j + min(proc % n , y)];
                    }
                }
        

            //Send sub-matrix to each process
            if (proc != MASTER)
            {
                int MPI_send_initial_subM = MPI_Send(&subM[0][0],
                        (subM_rows + x_sup + 2) * (subM_cols + y_sup + 2),
                        MPI_INT,
                        proc,
                        0,
                        MPI_COMM_WORLD);
                // abort if error in sending
                if (MPI_send_initial_subM != MPI_SUCCESS) {
                    MPI_Abort(MPI_COMM_WORLD, MPI_send_initial_subM);
                }
            }
        }
    }

    // Logic for task with non-Master ranks.
    else if (rank != MASTER) {
        // Receive initial sub-matrix from master process
        int MPI_recv_initial_subM = MPI_Recv(&subM[0][0],
         (subM_rows + x_sup + 2) * (subM_cols + y_sup + 2),
         MPI_INT,
         MASTER,
         0,
         MPI_COMM_WORLD,
         &Stat);
        // abort if error in receiving
        if (MPI_recv_initial_subM != MPI_SUCCESS) {
            MPI_Abort(MPI_COMM_WORLD, MPI_recv_initial_subM);
        }
    }

    int *neighbours = get_nbours(rank, m, n);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double simulation_start = MPI_Wtime();

    // Run the simulation for niter iterations
    for (iter = 0; iter < niter; iter++)
    {
        exchange_ghost_cells(&subM[0][0], subM_rows+x_sup+2, subM_cols+y_sup+2, neighbours);
        
        //update sub-matrix
        int temp[subM_rows+x_sup+2][subM_cols+y_sup+2]= {};
        for (i = 0; i < subM_rows+x_sup; i++)
        {
            for (j = 0; j < subM_cols+y_sup; j++)
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
        memcpy(subM, temp, sizeof(int) * (subM_rows + x_sup+2) * (subM_cols + y_sup+2));

        if ((iter+1) % 10 == 0) {       // TODO: change to 10 or whatever
            int local_live = calculate_number_live_cells(&subM[0][0], subM_rows+x_sup+2, subM_cols+y_sup+2);
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
            int x_sup=0, y_sup=0;
            if (proc/n < x) // check if extra row needed
            {
                x_sup = 1;
            }
            if (proc%n < y) // check if extra column needed
            {
                y_sup = 1;
            }

            if (proc == MASTER) {
                // Copy its own sub-matrix into the correct position in M
                for (i = 0; i < subM_rows+ x_sup; i++)
                {
                    for (j = 0; j < subM_cols+ y_sup; j++)
                    {           
                        M[i][j] = subM[i+1][j+1];
                    }
                }
            } else {
                // Receive final sub-matrices from other processes
                int MPI_Recv_final_subM = MPI_Recv(&subM[0][0],
                         (subM_rows + x_sup + 2) * (subM_cols + y_sup + 2),
                         MPI_INT,
                         proc,
                         3,
                         MPI_COMM_WORLD,
                         &Stat);
                // abort if error in receiving
                if (MPI_Recv_final_subM != MPI_SUCCESS) {
                    MPI_Abort(MPI_COMM_WORLD, MPI_Recv_final_subM);
                }
                
                // Copy received sub-matrix into the correct position in M
                for (i = 0; i < subM_rows; i++)
                {
                    for (j = 0; j < subM_cols; j++)
                    {           
                        M[(proc/n)*subM_rows + min(proc/n, x) + i][(proc%n)*subM_cols + min(proc%n, y) + j] = subM[i+1][j+1];
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
        // Non-master processes send their final sub-matrix back to the master process
        int MPI_Send_final_subM = MPI_Send(
            &subM[0][0],
            (subM_rows + x_sup + 2) * (subM_cols + y_sup + 2),
            MPI_INT,
            MASTER,
            3,
            MPI_COMM_WORLD
        );
        // abort if error in sending
        if (MPI_Send_final_subM != MPI_SUCCESS) {
            MPI_Abort(MPI_COMM_WORLD, MPI_Send_final_subM);
        }
    }

    // Terminates the MPI environment. Must be the last MPI call in the program.
    free(neighbours);
    MPI_Finalize();
}
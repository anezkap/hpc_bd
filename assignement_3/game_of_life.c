/******************************************************************************
* FILE: game_of_life.c
* DESCRIPTION: 
*   Full script for Game of Life
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int *get_nbours(int pid, int m, int n);
void exchange_ghost_cells(int *matrix, int num_rows, int num_cols, int *neighbours);
int calculate_number_live_cells(int *matrix, int num_rows, int num_cols);
// void get_total_live_cells(int *matrix, int num_rows, int num_cols, MPI_Comm comm)

void print_matrix(int *matrix, int rows, int cols, int rank, const char *label) {
  printf("\n[Rank %d] %s:\n", rank, label);
  for (int i = 0; i < rows; i++) {
    printf("[Rank %d] ", rank);
    for (int j = 0; j < cols; j++) {
      printf("%3d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  int numtasks, rank;
  int *neighbours;

  int num_iters = 50; // Number of iterations for the simulation

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    printf("Total number of processes: %d.\n", numtasks);
  }

  // Initialize matrix

  // Distribute submatrices to each process

  // Get neighbours for each process
  neighbours = get_nbours(rank, m, n);

  for (int i = 0; i < num_iters; i++) {
    exchange_ghost_cells(matrix, matrix_rows, matrix_cols, neighbours);

    // update_matrix(matrix, matrix_rows, matrix_cols); // Implement the Game of Life rules here

    // Print number of live cells after each 10 iterations
    if (i % 10 == 0) {
        int local_live = calculate_number_live_cells(matrix, matrix_rows, matrix_cols);
        int global_live = 0;
    
        MPI_Reduce(&local_live, &global_live, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
        if (rank == 0) {
            printf("[Iteration %d] Total live cells = %d\n", i, global_live);
        }
    }
  }

  // Synchronize before printing summary
  MPI_Barrier(MPI_COMM_WORLD);

  int local_live = calculate_number_live_cells(matrix, matrix_rows, matrix_cols);
  int global_live = 0;

  MPI_Reduce(&local_live, &global_live, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\n=== Simulation completed ===\n");
    printf("[Iteration %d] Total live cells = %d\n", i, global_live);
  }

  free(neighbours);
  MPI_Finalize();
  return 0;
}

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
  int m = 2, n = 2;  // 2x2 grid of processes
  int data_rows = 2, data_cols = 2;
  int matrix_rows = 4, matrix_cols = 4;  // data + ghost cells
  int *matrix;
  int *neighbours;

  int num_iters = 50; // Number of iterations for the simulation

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (numtasks != 4) {
    if (rank == 0) {
      printf("This test requires exactly 4 processes. Got %d.\n", numtasks);
    }
    MPI_Finalize();
    return 1;
  }

  // Allocate and initialize matrix
  // Layout: 4x4 matrix with ghost cells on borders
  // [0,0] [0,1] [0,2] [0,3]
  // [1,0] [1,1] [1,2] [1,3]
  // [2,0] [2,1] [2,2] [2,3]
  // [3,0] [3,1] [3,2] [3,3]
  // Where rows/cols 0 and 3 are ghost cells
  matrix = (int *)malloc(matrix_rows * matrix_cols * sizeof(int));

  // Initialize matrix: each element = rank*100 + row*10 + col
  // This makes it easy to identify the original value and detect incorrect exchanges
  for (int i = 0; i < matrix_rows; i++) {
    for (int j = 0; j < matrix_cols; j++) {
      matrix[i * matrix_cols + j] = rank * 100 + i * 10 + j;
    }
  }

  // Get neighbours for this rank in 2x2 grid
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

  if (rank == 0) {
    printf("\n=== Simulation completed ===\n");
  }

  free(matrix);
  free(neighbours);
  MPI_Finalize();
  return 0;
}

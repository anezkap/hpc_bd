/******************************************************************************
* FILE: ghost_exchange_neighbours.c
* DESCRIPTION: 
*   This program has two functions:
*   1) Get the neighbouring ranks of each process in a 2D Cartesian topology.
*   2) Exchange ghost cells with their neighbours using MPI derived datatypes.
*   3) Calculate the number of live cells in a given matrix (excluding ghost cells).
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * get_nbours: Get the neighbouring ranks of a process in a 2D grid
 * @param pid: Process ID (rank)
 * @param m: Number of process rows in the grid
 * @param n: Number of process columns in the grid
 * @return: Pointer to array of 4 neighbours [left, right, top, bottom], or NULL on error
 */
int *get_nbours(int pid, int m, int n)
{
  int *neighbours = (int *)malloc(4 * sizeof(int));
  
  if (neighbours == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failed in get_nbours()\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return NULL;
  }
  
  // Left, right, top, bottom
  neighbours[0] = (pid % n == 0) ? -1 : pid - 1;
  neighbours[1] = (pid % n == n - 1) ? -1 : pid + 1;
  neighbours[2] = (pid < n) ? -1 : pid - n;
  neighbours[3] = (pid >= (m - 1) * n) ? -1 : pid + n;

  return neighbours;
}

void send_side_columns(int *matrix, int num_rows, int num_cols, int *neighbours, MPI_Request *reqs, int *r)
{
  // Create MPI derived datatype for side columns and send to neighbours
  int array_of_sizes[2] = {num_rows, num_cols};
  int array_of_subsizes[2] = {num_rows - 2, 1};
  int array_of_starts[2];
  array_of_starts[0] = 1;
  MPI_Datatype column_type;

  if (neighbours[0] != -1) {
    // Send column 1 to left neighbour
    array_of_starts[1] = 1;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                              array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Isend(matrix, 1, column_type, neighbours[0], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }

  if (neighbours[1] != -1) {
    // Send column num_cols-2 to right neighbour
    array_of_starts[1] = num_cols - 2;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                              array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Isend(matrix, 1, column_type, neighbours[1], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }
}

void receive_side_columns(int *matrix, int num_rows, int num_cols, int *neighbours, MPI_Request *reqs, int *r)
{
  // Create MPI derived datatype for side columns and receive from neighbours
  int array_of_sizes[2] = {num_rows, num_cols};
  int array_of_subsizes[2] = {num_rows - 2, 1};
  int array_of_starts[2];
  MPI_Datatype column_type;

  if (neighbours[0] != -1) {
    // Receive from left neighbour into column 0 (ghost column)
    array_of_starts[0] = 1;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Irecv(matrix, 1, column_type, neighbours[0], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }

  if (neighbours[1] != -1) {
    // Receive from right neighbour into column num_cols-1 (ghost column)
    array_of_starts[0] = 1;
    array_of_starts[1] = num_cols - 1;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Irecv(matrix, 1, column_type, neighbours[1], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }
}

void receive_bottom_top_columns(int *matrix, int num_rows, int num_cols, int *neighbours, MPI_Request *reqs, int *r)
{
  // Create MPI derived datatype for top and bottom rows and receive from neighbours
  int array_of_sizes[2] = {num_rows, num_cols};
  int array_of_subsizes[2] = {1, num_cols};
  int array_of_starts[2];
  MPI_Datatype column_type;

  if (neighbours[2] != -1) {
    // Receive from top neighbour into row 0 (ghost row)
    array_of_starts[0] = 0;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Irecv(matrix, 1, column_type, neighbours[2], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }

  if (neighbours[3] != -1) {
    // Receive from bottom neighbour into row num_rows-1 (ghost row)
    array_of_starts[0] = num_rows - 1;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Irecv(matrix, 1, column_type, neighbours[3], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }
}

void send_bottom_top_columns(int *matrix, int num_rows, int num_cols, int *neighbours, MPI_Request *reqs, int *r)
{
  // Create MPI derived datatype for top and bottom rows and send to neighbours
  int array_of_sizes[2] = {num_rows, num_cols};
  int array_of_subsizes[2] = {1, num_cols};
  int array_of_starts[2];
  MPI_Datatype column_type;

  if (neighbours[2] != -1) {
    // Send first data row (row 1) to top neighbour
    array_of_starts[0] = 1;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Isend(matrix, 1, column_type, neighbours[2], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }

  if (neighbours[3] != -1) {
    // Send last data row (row num_rows-2) to bottom neighbour
    array_of_starts[0] = num_rows - 2;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Isend(matrix, 1, column_type, neighbours[3], 1, MPI_COMM_WORLD, &reqs[(*r)++]);
    MPI_Type_free(&column_type);
  }
}

/**
 * exchange_ghost_cells: Exchange ghost cells with neighbouring processes in a 2D grid
 * @param matrix: Pointer to the (sub)matrix data (includes ghost cells)
 * @param num_rows: Number of rows in the local matrix (including ghost cells)
 * @param num_cols: Number of columns in the local matrix (including ghost cells)
 * @param neighbours: Array of 4 neighbouring process ranks [left, right, top, bottom]
 */
void exchange_ghost_cells(int *matrix, int num_rows, int num_cols, int *neighbours)
{
  // Create requests array and counter
  MPI_Request reqs[8];
  int r = 0;
  int mpi_err;

  // 1. Receive side ghost columns
  receive_side_columns(matrix, num_rows, num_cols, neighbours, reqs, &r);

  // 2. Send side columns
  send_side_columns(matrix, num_rows, num_cols, neighbours, reqs, &r);

  // 3. Receive top and bottom rows
  receive_bottom_top_columns(matrix, num_rows, num_cols, neighbours, reqs, &r);

  // Wait only for side receives (if we have these neighbours), the other sends and receives can overlap
  if (neighbours[0] != -1 && neighbours[1] != -1) {
    mpi_err = MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) {
      fprintf(stderr, "ERROR: MPI_Wait failed\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    mpi_err = MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) {
      fprintf(stderr, "ERROR: MPI_Wait failed\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
  else if (neighbours[0] != -1 || neighbours[1] != -1) {
    mpi_err = MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) {
      fprintf(stderr, "ERROR: MPI_Wait failed\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // 4. Send top and bottom rows
  send_bottom_top_columns(matrix, num_rows, num_cols, neighbours, reqs, &r);

  // 5. Wait for all remaining requests to complete
  mpi_err = MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);
  if (mpi_err != MPI_SUCCESS) {
    fprintf(stderr, "ERROR: MPI_Waitall failed\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
}


/**
 * calculate_number_live_cells: Count the number of live cells in a matrix
 * @param matrix: Pointer to the (sub)matrix data (includes ghost cells)
 * @param num_rows: Number of rows in the (sub)matrix (including ghost cells)
 * @param num_cols: Number of columns in the (sub)matrix (including ghost cells)
 * @return: Number of live cells in the non-ghost region, or 0 if matrix is NULL
 */
int calculate_number_live_cells(int *matrix, int num_rows, int num_cols)
{
  // Returns the number of live cells in the matrix excluding ghost cells
  int live_cells = 0;

  if (matrix == NULL) {
    fprintf(stderr, "ERROR: NULL matrix pointer in calculate_number_live_cells()\n");
    return 0;
  }

  for (int i = 1; i < num_rows - 1; i++) {
    for (int j = 1; j < num_cols - 1; j++) {
      if (matrix[i * num_cols + j] == 1) {
        live_cells++;
      }
    }
  }

  return live_cells;
}
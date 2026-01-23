/******************************************************************************
* FILE: ghost_exchange_neighbours.c
* DESCRIPTION: 
*   This program has two functions:
*   1) Get the neighbouring ranks of each process in a 2D Cartesian topology.
*   2) Exchange ghost cells with these neighbouring ranks.
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int *get_nbours (int pid, int m, int n)
{
  int *neighbours = (int *)malloc(4 * sizeof(int));
  
  // Left, right, top, bottom
  neighbours[0] = (pid % n == 0) ? -1 : pid - 1;
  neighbours[1] = (pid % n == n - 1) ? -1 : pid + 1;
  neighbours[2] = (pid < n) ? -1 : pid - n;
  neighbours[3] = (pid >= (m - 1) * n) ? -1 : pid + n;

  return neighbours;
}


void exchange_ghost_cells(int *matrix, int num_rows, int num_cols, int *neighbours)
{
  // Create subarray type for column (excluding first and last rows)
  int array_of_sizes[2] = {num_rows, num_cols};
  int array_of_subsizes[2] = {num_rows - 2, 1};
  int array_of_starts[2];
  MPI_Datatype column_type;
  MPI_Status status;

  // 1. Send left and right columns
  if (neighbours[0] != -1) {
    // Send second column (column 1) to left neighbour
    array_of_starts[0] = 1;
    array_of_starts[1] = 1;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Send(matrix, 1, column_type, neighbours[0], 0, MPI_COMM_WORLD);
    MPI_Type_free(&column_type);
  }

  if (neighbours[1] != -1) {
    // Send second-to-last column (column num_cols-2) to right neighbour
    array_of_starts[0] = 1;
    array_of_starts[1] = num_cols - 2;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Send(matrix, 1, column_type, neighbours[1], 0, MPI_COMM_WORLD);
    MPI_Type_free(&column_type);
  }

  // 2. Receive left and right ghost columns
  if (neighbours[0] != -1) {
    // Receive from left neighbour into column 0 (ghost column)
    array_of_starts[0] = 1;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Recv(matrix, 1, column_type, neighbours[0], 0, MPI_COMM_WORLD, &status);
    MPI_Type_free(&column_type);
  }

  if (neighbours[1] != -1) {
    // Receive from right neighbour into column num_cols-1 (ghost column)
    array_of_starts[0] = 1;
    array_of_starts[1] = num_cols - 1;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Recv(matrix, 1, column_type, neighbours[1], 0, MPI_COMM_WORLD, &status);
    MPI_Type_free(&column_type);
  }

  // 3. Send and receive top and bottom rows
  array_of_subsizes[0] = 1;
  array_of_subsizes[1] = num_cols;

  if (neighbours[2] != -1) {
    // Send first data row (row 1) to top neighbour
    array_of_starts[0] = 1;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Send(matrix, 1, column_type, neighbours[2], 0, MPI_COMM_WORLD);
    MPI_Type_free(&column_type);
  }

  if (neighbours[3] != -1) {
    // Send last data row (row num_rows-2) to bottom neighbour
    array_of_starts[0] = num_rows - 2;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Send(matrix, 1, column_type, neighbours[3], 0, MPI_COMM_WORLD);
    MPI_Type_free(&column_type);
  }

  if (neighbours[2] != -1) {
    // Receive from top neighbour into row 0 (ghost row)
    array_of_starts[0] = 0;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Recv(matrix, 1, column_type, neighbours[2], 0, MPI_COMM_WORLD, &status);
    MPI_Type_free(&column_type);
  }

  if (neighbours[3] != -1) {
    // Receive from bottom neighbour into row num_rows-1 (ghost row)
    array_of_starts[0] = num_rows - 1;
    array_of_starts[1] = 0;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, 
                             array_of_starts, MPI_ORDER_C, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    MPI_Recv(matrix, 1, column_type, neighbours[3], 0, MPI_COMM_WORLD, &status);
    MPI_Type_free(&column_type);
  }
} 


// int main (int argc, char *argv[])
// {
// int numtasks, rank, dest, tag, source, rc, count;
// char inmsg, outmsg='x';
// MPI_Status Stat;

// MPI_Init(&argc,&argv);
// MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// printf("Task %d starting...\n",rank);

// if (rank == 0) {
//   if (numtasks > 2) 
//     printf("Numtasks=%d. Only 2 needed. Ignoring extra...\n",numtasks);
//   dest = rank + 1;
//   source = dest;
//   tag = rank;
//   rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
//   printf("Sent to task %d...\n",dest);
//   rc = MPI_Recv(&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
//   printf("Received from task %d...\n",source);
//   }

// else if (rank == 1) {
//   dest = rank - 1;
//   source = dest;
//   tag = rank;
//   rc = MPI_Recv(&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
//   printf("Received from task %d...\n",source);
//   rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
//   printf("Sent to task %d...\n",dest);
//   }

// if (rank < 2) {
//   rc = MPI_Get_count(&Stat, MPI_CHAR, &count);
//   printf("Task %d: Received %d char(s) from task %d with tag %d \n",
//          rank, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
//   }

// MPI_Finalize();
// }
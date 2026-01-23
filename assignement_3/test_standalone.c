#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int *get_nbours (int pid, int m, int n)
{
  int *neighbours = (int *)malloc(4 * sizeof(int));
  
  // neighbours[0] = left, neighbours[1] = right
  // neighbours[2] = up, neighbours[3] = down
  
  // Left: same row, column - 1
  neighbours[0] = (pid % n == 0) ? -1 : pid - 1;
  
  // Right: same row, column + 1
  neighbours[1] = (pid % n == n - 1) ? -1 : pid + 1;
  
  // Up: row - 1, same column
  neighbours[2] = (pid < n) ? -1 : pid - n;
  
  // Down: row + 1, same column
  neighbours[3] = (pid >= (m - 1) * n) ? -1 : pid + n;

  return neighbours;
}

void test_get_nbours() {
  // Test case 1: 2x3 grid, process 0 (top-left corner)
  int *neighbours = (int *)get_nbours(0, 2, 3);
  assert(neighbours[0] == -1);  // left
  assert(neighbours[1] == 1);   // right
  assert(neighbours[2] == -1);  // up
  assert(neighbours[3] == 3);   // down
  free(neighbours);
  printf("Test 1 passed: top-left corner\n");

  // Test case 2: 2x3 grid, process 5 (bottom-right corner)
  neighbours = (int *)get_nbours(5, 2, 3);
  assert(neighbours[0] == 4);   // left
  assert(neighbours[1] == -1);  // right
  assert(neighbours[2] == 2);   // up
  assert(neighbours[3] == -1);  // down
  free(neighbours);
  printf("Test 2 passed: bottom-right corner\n");

  // Test case 3: 3x3 grid, process 4 (center)
  neighbours = (int *)get_nbours(4, 3, 3);
  assert(neighbours[0] == 3);   // left
  assert(neighbours[1] == 5);   // right
  assert(neighbours[2] == 1);   // up
  assert(neighbours[3] == 7);   // down
  free(neighbours);
  printf("Test 3 passed: center process\n");

  // Test case 4: 2x2 grid, process 1 (top-right corner)
  neighbours = (int *)get_nbours(1, 2, 2);
  assert(neighbours[0] == 0);   // left
  assert(neighbours[1] == -1);  // right
  assert(neighbours[2] == -1);  // up
  assert(neighbours[3] == 3);   // down
  free(neighbours);
  printf("Test 4 passed: top-right corner\n");

  printf("All tests passed!\n");
}

int main() {
  test_get_nbours();
  return 0;
}

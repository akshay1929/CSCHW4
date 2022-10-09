#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   int blocks = n/block_size;
   double *aLoc = new double[block_size*block_size];
   double *bLoc = new double[block_size*block_size];
   double *cLoc = new double[block_size*block_size];
   // x = i
   // y = j
   // z = k
   // i = y
   // j = z
   // k = x
   int blocks = n/block_size;
   for (int x = 0; x < blocks; x++) {
      for (int y = 0; y < blocks; y++) {
         //Copy to cLoc
         for (int cj = 0; cj < block_size; cj++) {
            for (int ci = 0; ci < block_size; ci++) {
               cLoc[block_size * cj + ci] = C[(block_size * y + cj) * n + block_size * x + ci];
            }
         }

         for (int z = 0; z < blocks; z++) {
            //Copy to aLoc and bLoc
            for (int cj = 0; cj < block_size; cj++) {
               for (int ci = 0; ci < block_size; ci++) {
                  aLoc[block_size * cj + ci] = A[(block_size * z + cj) * n + block_size * x + ci];
                  bLoc[block_size * cj + ci] = B[(block_size * y + cj) * n + block_size * z + ci];
               }
            }
            //Multiplication
            for (int i = 0; i < n; i++) {
               for (int j = 0; j < n; j++) {
                  for (int k = 0; k < n; k++) {
                     cLoc[i * block_size + k] += aLoc[j * block_size + k] * bLoc[i * block_size + j];
                  }
               }
            }
         }
         //Copy back to C
         for (int cj = 0; cj < block_size; cj++) {
            for (int ci = 0; ci < block_size; ci++) {
               C[(block_size * y + ci) * n + block_size * x + ci] = cLoc[block_size * cj + ci];
            }
         }
      }
   }
}

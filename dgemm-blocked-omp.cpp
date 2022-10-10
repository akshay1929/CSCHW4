#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


void copytoNew(double* newMatrix, double *old, int n, int block_size, int i, int j) {
   for (int x = 0; x < block_size; x++) {
      for (int y = 0; y < block_size; y++) {
         newMatrix[x * block_size + y] = old[(j * block_size + x) * n + i * block_size + y];
      }
   }
}
void copytoOld(double* newMatrix, double *old, int n, int block_size, int i, int j) {
   for (int x = 0; x < block_size; x++) {
      for (int y = 0; y < block_size; y++) {
         newMatrix[(j * block_size + x) * n + i * block_size + y] = old[x * block_size + y];
      }
   }
}

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
   for (int i = 0; i < blocks; i++) {
      for (int j = 0; j < blocks; j++) {
         //Makes copy of block from C
         copytoNew(cLoc, C, n, block_size, i, j);

         for (int k = 0; k < blocks; k++) {
            //Makes copy of block from A
               copytoNew(aLoc, A, n, block_size, i, k);
               //Makes copy of block from B
               copytoNew(bLoc, B, n, block_size, k, j);

            for (int a = 0; a < block_size; a++) {
               for (int b = 0; b < block_size; b++) {
                  for (int c = 0; c < block_size; c++) {
                     cLoc[a * block_size + c] += aLoc[b * block_size + c] * bLoc[a * block_size + b];
                  }
               }
            }
         }
         //Makes copy of block to C
         copytoOld(C, cLoc, n, block_size, i, j);
      }
   }
   
   delete[] aLoc;
   delete[] bLoc;
   delete[] cLoc;
}

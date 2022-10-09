#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


void copytoNew(double* newMatrix, double *old, int n, int block_size, int x, int y) {
   for (int cj = 0; cj < block_size; cj++) {
      for (int ci = 0; ci < block_size; ci++) {
         newMatrix[block_size * cj + ci] = old[(block_size * y + cj) * n + block_size * x + ci];
      }
   }
}
void copytoOld(double* newMatrix, double *old, int n, int block_size, int x, int y) {
   for (int cj = 0; cj < block_size; cj++) {
      for (int ci = 0; ci < block_size; ci++) {
         old[(block_size * y + ci) * n + block_size * x + ci] = newMatrix[block_size * cj + ci];
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
   // x = i
   // y = j
   // z = k
   // i = y
   // j = z
   // k = x
   for (int x = 0; x < blocks; x++) {
      for (int y = 0; y < blocks; y++) {
         //Copy to cLoc
         //copytoNew(cLoc, C, n, block_size, x, y);
         for (int cj = 0; cj < block_size; cj++) {
            for (int ci = 0; ci < block_size; ci++) {
               cLoc[block_size * cj + ci] = C[(block_size * y + cj) * n + block_size * x + ci];
            }
         }

         for (int z = 0; z < blocks; z++) {
            //Copy to aLoc and bLoc
            // copytoNew(aLoc, A, n, block_size, x, z);
            // copytoNew(bLoc, B, n, block_size, y, z);
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
         //copytoOld(C, cLoc, n, block_size, x, y);
         for (int cj = 0; cj < block_size; cj++) {
            for (int ci = 0; ci < block_size; ci++) {
               C[(block_size * y + ci) * n + block_size * x + ci] = cLoc[block_size * cj + ci];
            }
         }
      }
   }
   // for (int a = 0; a < n; a += block_size) {
   //    for (int b = 0; b < n; b += block_size) {
   //       for (int c = 0; c < n; c += block_size) { 
   //          LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
   //          #pragma parallel for
   //          for (int i = a; i < a + block_size; i++) {
   //             for (int j = b; j < b + block_size; j++) {
   //                for (int k = c; k < c + block_size; k++) {
   //                   C[j * n + i] += A[k * n + i] * B[j * n + k];
   //                }
   //             }
   //          }
   //          LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   //       }
   //    }
   // }
}

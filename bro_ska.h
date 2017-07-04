/*
 * Generic multiplication (+ add) routines for matrices smaller than 128x128
 * For legacy systems that don't have SSE2
 * 
 * Pierre Karpman (CWI)
 * 2017-07
 */

#include <stdint.h>
#include <m4ri/m4ri.h>

#ifndef __MUL_BRO_SKA_H
#define __MUL_BRO_SKA_H

/*
 * All the following do:
 * if (clear) res = V*A
 * if (!clear) res = res + V*A
 */


/*
 * assert(V->ncols <= 64)
 * assert(A->ncols <= 64)
 */
void mul_64_64_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear);

/*
 * assert(V->ncols > 64)
 * assert(V->ncols <= 128)
 * assert(A->ncols <= 64)
 */
void mul_128_64_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear);

/*
 * assert(V->ncols <= 64)
 * assert(A->ncols > 64)
 * assert(A->ncols <= 128)
 */
void mul_64_128_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear);

/*
 * assert(V->ncols > 64)
 * assert(V->ncols <= 128)
 * assert(A->ncols > 64)
 * assert(A->ncols <= 128)
 */
void mul_128_128_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear);

#endif

/*
 * ``Fast'' matrix multiplication in GF(2) for small dimensions
 * Uses broadcast-based vectorized algorithms
 * (See e.g. (Käsper and Schwabe, 2009) and (Augot et al., 2014) for illustrations)
 *
 * Pierre Karpman
 * 2017-06--07
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <immintrin.h>
#include <m4ri/m4ri.h>
#include "hc128rand.h"

#include "bro_ska.h"

/*
 * For custom randomization thru callback
 */
uint64_t my_little_rand(void *willignore)
{
	return (((uint64_t)(hc128random()) << 32) ^ ((uint64_t)hc128random()));
}

/* no full checks yet */
void mul_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;

	void(*bros[4])(mzd_t*, mzd_t*, mzd_t*, int) = {&mul_64_64_bro_ska, &mul_128_64_bro_ska, &mul_64_128_bro_ska, &mul_128_128_bro_ska};
	unsigned fun = (((acols-1)/64)<<1)|((vcols-1)/64);
	
	return bros[fun](res, V, A, clear);

//	if ((acols <= 64) && (vcols <= 64))
//		return mul_64_64_bro_ska(res, V, A, clear);
//	if (acols <= 64)
//		return mul_128_64_bro_ska(res, V, A, clear);
//	if (vcols <= 64)
//		return mul_64_128_bro_ska(res, V, A, clear);
//	return mul_128_128_bro_ska(res, V, A, clear);
}


/* TODO Waiting to be moved elsewhere */

///* = 32 = */
//
///*
// * Same as above, but using SSE4.1
// * (lower version could be used in a rather similar way)
// */
//void mul_va3232_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
////	assert(vcols <= 32);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
////		uint64_t v = mzd_read_bits(V, j, 0, vcols);
//		uint64_t v = *(V->rows[j]); // FIXME: always correct?
//
//		__m128i m2, rr;
//		__m128i acc = _mm_setzero_si128();
//		__m128i m1  = _mm_set_epi32(1, 2, 4, 8);
//		__m128i vv  = _mm_set1_epi32(v);
//
//		int i;
//		/* main loop */
//		for (i = 0; i < (int)(vcols-4); i+=4)
//		{
////			rr  = _mm_set_epi32(mzd_read_bits(A, i, 0, acols), mzd_read_bits(A, i+1, 0, acols), mzd_read_bits(A, i+2, 0, acols), mzd_read_bits(A, i+3, 0, acols));
//			rr  = _mm_set_epi32(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3])); // FIXME: always correct?
//			m2  = _mm_and_si128(vv, m1);
//			m2  = _mm_cmpeq_epi32(m1, m2);
//			rr  = _mm_and_si128(rr, m2);
//			acc = _mm_xor_si128(acc, rr);
//			m1  = _mm_slli_epi32(m1, 4);
//		}
//		/* footer */
////		uint32_t r0 = mzd_read_bits(A, i, 0, acols);
////		uint32_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
////		uint32_t r2 = i+2 >= arows ? 0 : mzd_read_bits(A, i+2, 0, acols);
////		uint32_t r3 = i+3 >= arows ? 0 : mzd_read_bits(A, i+3, 0, acols);
//		uint32_t r0 = *(A->rows[i]); // FIXME: always correct?
//		uint32_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME: always correct?
//		uint32_t r2 = i+2 >= arows ? 0 : *(A->rows[i+2]); // FIXME: always correct?
//		uint32_t r3 = i+3 >= arows ? 0 : *(A->rows[i+3]); // FIXME: always correct?
//		rr  = _mm_set_epi32(r0, r1, r2, r3);
//		m2  = _mm_and_si128(vv, m1);
//		m2  = _mm_cmpeq_epi32(m1, m2);
//		rr  = _mm_and_si128(rr, m2);
//		acc = _mm_xor_si128(acc, rr);
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, acols, 0);
//		}
//		uint32_t tmpacc1 = _mm_extract_epi32(acc, 0) ^ _mm_extract_epi32(acc, 1);
//		uint32_t tmpacc2 = _mm_extract_epi32(acc, 2) ^ _mm_extract_epi32(acc, 3);
//		tmpacc1 ^= tmpacc2;
//		mzd_xor_bits(res, j, 0, acols, tmpacc1);
//	}
//
//	return;
//}
//
///*
// * Same as above, but using AVX2
// */
//void mul_va3232_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
////	assert(vcols <= 32);
////	assert(arows == vcols);
//
//
//	for (int j = 0; j < vrows; j++)
//	{
////		uint64_t v = mzd_read_bits(V, j, 0, vcols);
//		uint64_t v = *(V->rows[j]); // FIXME: always correct?
//
//		__m256i m2, rr;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set_epi32(1, 2, 4, 8, 16, 32, 64, 128);
//		__m256i vv  = _mm256_set1_epi32(v);
//
//		/* main loop */
//		int i;
//		for (i = 0; i < (int)(vcols-8); i+=8)
//		{
////			rr  = _mm256_set_epi32(mzd_read_bits(A, i, 0, acols), mzd_read_bits(A, i+1, 0, acols), mzd_read_bits(A, i+2, 0, acols), mzd_read_bits(A, i+3, 0, acols),
////					mzd_read_bits(A, i+4, 0, acols), mzd_read_bits(A, i+5, 0, acols), mzd_read_bits(A, i+6, 0, acols), mzd_read_bits(A, i+7, 0, acols));
//			rr  = _mm256_set_epi32(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3]), *(A->rows[i+4]), *(A->rows[i+5]), *(A->rows[i+6]), *(A->rows[i+7])); // FIXME: always correct?
//			m2  = _mm256_and_si256(vv, m1);
//			m2  = _mm256_cmpeq_epi32(m1, m2);
//			rr  = _mm256_and_si256(rr, m2);
//			acc = _mm256_xor_si256(acc, rr);
//			m1  = _mm256_slli_epi32(m1, 8);
//		}
//		/* footer */
////		uint32_t r0 = mzd_read_bits(A, i, 0, acols);
////		uint32_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
////		uint32_t r2 = i+2 >= arows ? 0 : mzd_read_bits(A, i+2, 0, acols);
////		uint32_t r3 = i+3 >= arows ? 0 : mzd_read_bits(A, i+3, 0, acols);
////		uint32_t r4 = i+4 >= arows ? 0 : mzd_read_bits(A, i+4, 0, acols);
////		uint32_t r5 = i+5 >= arows ? 0 : mzd_read_bits(A, i+5, 0, acols);
////		uint32_t r6 = i+6 >= arows ? 0 : mzd_read_bits(A, i+6, 0, acols);
////		uint32_t r7 = i+7 >= arows ? 0 : mzd_read_bits(A, i+7, 0, acols);
//		uint32_t r0 = *(A->rows[i]); // FIXME: always correct?
//		uint32_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME: always correct?
//		uint32_t r2 = i+2 >= arows ? 0 : *(A->rows[i+2]); // FIXME: always correct?
//		uint32_t r3 = i+3 >= arows ? 0 : *(A->rows[i+3]); // FIXME: always correct?
//		uint32_t r4 = i+4 >= arows ? 0 : *(A->rows[i+4]); // FIXME: always correct?
//		uint32_t r5 = i+5 >= arows ? 0 : *(A->rows[i+5]); // FIXME: always correct?
//		uint32_t r6 = i+6 >= arows ? 0 : *(A->rows[i+6]); // FIXME: always correct?
//		uint32_t r7 = i+7 >= arows ? 0 : *(A->rows[i+7]); // FIXME: always correct?
//		rr  = _mm256_set_epi32(r0, r1, r2, r3, r4, r5, r6, r7);
//		m2  = _mm256_and_si256(vv, m1);
//		m2  = _mm256_cmpeq_epi32(m1, m2);
//		rr  = _mm256_and_si256(rr, m2);
//		acc = _mm256_xor_si256(acc, rr);
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, acols, 0);
//		}
//		uint32_t tmpacc1 = _mm256_extract_epi32(acc, 0) ^ _mm256_extract_epi32(acc, 1);
//		uint32_t tmpacc2 = _mm256_extract_epi32(acc, 2) ^ _mm256_extract_epi32(acc, 3);
//		uint32_t tmpacc3 = _mm256_extract_epi32(acc, 4) ^ _mm256_extract_epi32(acc, 5);
//		uint32_t tmpacc4 = _mm256_extract_epi32(acc, 6) ^ _mm256_extract_epi32(acc, 7);
//		tmpacc1 ^= tmpacc2;
//		tmpacc3 ^= tmpacc4;
//		tmpacc1 ^= tmpacc3;
//		mzd_xor_bits(res, j, 0, acols, tmpacc1);
//	}
//
//	return;
//}
//
///* = 64 */
//
///*
// * Same as above, but using SSE4.1
// * (lower version could be used in a rather similar way)
// */
//void mul_va6464_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
////	assert(vcols <= 64);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
////		uint64_t v = mzd_read_bits(V, j, 0, vcols);
//		uint64_t v = *(V->rows[j]); // FIXME: always correct?
//
//		__m128i m2, rr;
//		__m128i acc = _mm_setzero_si128();
//		__m128i m1  = _mm_set_epi64x(1, 2);
//		__m128i vv  = _mm_set1_epi64x(v);
//
//		int i;
//		/* main loop */
//		for (i = 0; i < (int)(vcols-2); i+=2)
//		{
////			rr  = _mm_set_epi64x(mzd_read_bits(A, i, 0, acols), mzd_read_bits(A, i+1, 0, acols));
//			rr  = _mm_set_epi64x(*(A->rows[i]),*(A->rows[i+1])); // FIXME: always correct?
//			m2  = _mm_and_si128(vv, m1);
//			m2  = _mm_cmpeq_epi64(m1, m2);
//			rr  = _mm_and_si128(rr, m2);
//			acc = _mm_xor_si128(acc, rr);
//			m1  = _mm_slli_epi64(m1, 2);
//		}
//		/* footer */
////		uint64_t r0 = mzd_read_bits(A, i, 0, acols);
////		uint64_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
//		uint64_t r0 = *(A->rows[i]); // FIXME: always correct?
//		uint64_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME : always correct?
//		rr  = _mm_set_epi64x(r0, r1);
////		rr = _mm_loadu_si128((__m128i*)A->rows[i]); // unfortunately, the matrix structure doesn't allow to do this much faster load
////		(not suitable for generic form anyway)
//		m2  = _mm_and_si128(vv, m1);
//		m2  = _mm_cmpeq_epi64(m1, m2);
//		rr  = _mm_and_si128(rr, m2);
//		acc = _mm_xor_si128(acc, rr);
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, acols, 0);
//		}
//		mzd_xor_bits(res, j, 0, acols, _mm_extract_epi64(acc, 0));
//		mzd_xor_bits(res, j, 0, acols, _mm_extract_epi64(acc, 1));
//	}
//
//	return;
//}
//
///*
// * Same as above, but using AVX2
// */
//void mul_va6464_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	//	assert(vcols <= 64);
//	//	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
////		uint64_t v = mzd_read_bits(V, j, 0, vcols);
//		uint64_t v = *(V->rows[j]); // FIXME: always correct?
//
//		__m256i m2, rr;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set_epi64x(1, 2, 4, 8);
//		__m256i vv  = _mm256_set1_epi64x(v);
//
//		int i;
//		/* main loop */
//		for (i = 0; i < (int)(vcols-4); i+=4)
//		{
////			rr  = _mm256_set_epi64x(mzd_read_bits(A, i, 0, 64), mzd_read_bits(A, i+1, 0, 64), mzd_read_bits(A, i+2, 0, 64), mzd_read_bits(A, i+3, 0, 64));
//			//		rr  = _mm256_loadu_si256((__m256i*)(A->rows[i])); // unfortunately, the matrix structure doesn't allow to do this much faster load
//			rr  = _mm256_set_epi64x(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3])); // FIXME: always correct?
//			//		ditto
//			m2  = _mm256_and_si256(vv, m1);
//			m2  = _mm256_cmpeq_epi64(m1, m2);
//			rr  = _mm256_and_si256(rr, m2);
//			acc = _mm256_xor_si256(acc, rr);
//			m1  = _mm256_slli_epi64(m1, 4);
//		}
//		/* footer */
////		uint64_t r0 = mzd_read_bits(A, i, 0, acols);
////		uint64_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
////		uint64_t r2 = i+2 >= arows ? 0 : mzd_read_bits(A, i+2, 0, acols);
////		uint64_t r3 = i+3 >= arows ? 0 : mzd_read_bits(A, i+3, 0, acols);
//		uint64_t r0 = *(A->rows[i]); // FIXME: always correct?
//		uint64_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME; always correct?
//		uint64_t r2 = i+2 >= arows ? 0 : *(A->rows[i+2]); // FIXME; always correct;
//		uint64_t r3 = i+3 >= arows ? 0 : *(A->rows[i+3]); // FIXME; always correct;
//		rr  = _mm256_set_epi64x(r0, r1, r2, r3);
//		m2  = _mm256_and_si256(vv, m1);
//		m2  = _mm256_cmpeq_epi64(m1, m2);
//		rr  = _mm256_and_si256(rr, m2);
//		acc = _mm256_xor_si256(acc, rr);
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, acols, 0);
//		}
//		uint64_t tmpacc1 = _mm256_extract_epi64(acc, 0) ^ _mm256_extract_epi64(acc, 1);
//		uint64_t tmpacc2 = _mm256_extract_epi64(acc, 2) ^ _mm256_extract_epi64(acc, 3);
//		tmpacc1 ^= tmpacc2;
//		mzd_xor_bits(res, j, 0, acols, tmpacc1);
//	}
//
//	return;
//}
//
///* = 128 = */
//
///*
// * All input allocated
// * if clear, reset res before adding the result
// * Computes VxA, where A is at most 128x128
// * Warning: also needs V to have > 64 columns (otherwise, a smaller implementation is selected)
// * Warning: also needs A to have > 64 columns (otherwise, a smaller implementation is selected)
// */
//void mul_va128128_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
////	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vcolhi = vcols - 64;
//	unsigned acolhi = acols - 64;
////	assert(vcols <= 128);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
////		__m128i v  = _mm_set_epi64x(mzd_read_bits(V, j, 64, vcolhi), mzd_read_bits(V, j, 0, 64));
//		__m128i v  = _mm_loadu_si128((__m128i*)V->rows[j]);
//
//		__m128i m2, m3, rr2, rr3;
//		__m128i acc = _mm_setzero_si128();
//		__m128i m1  = _mm_set1_epi64x(1);
//
//		for (int i = 0; i < 64; i++)
//		{
////			rr2  = _mm_set_epi64x(mzd_read_bits(A, i, 64, acolhi), mzd_read_bits(A, i, 0, 64));
//			rr2 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct?
//			m2  = _mm_and_si128(v, m1);
//			m2  = _mm_cmpeq_epi64(m1, m2);
//			if (i < vcolhi)
//			{
////				rr3 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, acolhi), mzd_read_bits(A, i+64, 0, 64));
//				rr3 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct?
//				m3  = _mm_shuffle_epi32(m2,0xFF);
//				rr3 = _mm_and_si128(rr3, m3);
//				acc = _mm_xor_si128(acc, rr3);
//			}
//			m2  = _mm_shuffle_epi32(m2,0x00);
//			rr2 = _mm_and_si128(rr2, m2);
//			acc = _mm_xor_si128(acc, rr2);
//			m1  = _mm_slli_epi64(m1, 1);
//		}
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, 64, 0);
//			mzd_and_bits(res, j, 64, acolhi, 0);
//		}
//		mzd_xor_bits(res, j, 0, 64, _mm_extract_epi64(acc, 0));
//		mzd_xor_bits(res, j, 64, acolhi, _mm_extract_epi64(acc, 1));
//	}
//
//	return;
//}
//
///* same not generic as above */
//void mul_va128128_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vcolhi = vcols - 64;
//	unsigned acolhi = acols - 64;
////	assert(vcols <= 128);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
////		__m128i v  = _mm_set_epi64x(mzd_read_bits(V, j, 64, vcolhi), mzd_read_bits(V, j, 0, 64));
//		__m128i v  = _mm_loadu_si128((__m128i*)V->rows[j]); // FIXME: always correct?
//
//		__m256i m2, m3, rr2, rr3;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set_epi64x(2, 2, 1, 1);
//		__m256i vv  = _mm256_set_m128i(v, v);
//		__m128i a0;
//		__m128i a1;
//
//		int i;
//		/* main loop */
//		for (i = 0; i < 62; i+=2)
//		{
////			a0 = _mm_set_epi64x(mzd_read_bits(A, i, 64, acolhi), mzd_read_bits(A, i, 0, 64));
////			a1 = _mm_set_epi64x(mzd_read_bits(A, i+1, 64, acolhi), mzd_read_bits(A, i+1, 0, 64));
//			a0 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct?
//			a1 = _mm_loadu_si128((__m128i*)A->rows[i+1]); // FIXME: always correct?
//
//			rr2 = _mm256_set_m128i(a1, a0);
//			m2  = _mm256_and_si256(vv, m1);
//			m2  = _mm256_cmpeq_epi64(m1, m2);
//			if (i < vcolhi)
//			{
////				a0 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, acolhi), mzd_read_bits(A, i+64, 0, 64));
//				a0 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct?
//				if (i+65 < arows)
//				{
////					a1 = _mm_set_epi64x(mzd_read_bits(A, i+65, 64, acolhi), mzd_read_bits(A, i+65, 0, 64));
//					a1 = _mm_loadu_si128((__m128i*)A->rows[i+65]); // FIXME: always correct?
//				}
//				else
//				{
//					a1 = _mm_setzero_si128();
//				}
//				rr3 = _mm256_set_m128i(a1, a0);
//				m3  = _mm256_shuffle_epi32(m2,0xFF);
//				rr3 = _mm256_and_si256(rr3, m3);
//				acc = _mm256_xor_si256(acc, rr3);
//			}
//			m2  = _mm256_shuffle_epi32(m2,0x00);
//			rr2 = _mm256_and_si256(rr2, m2);
//			acc = _mm256_xor_si256(acc, rr2);
//			m1  = _mm256_slli_epi64(m1, 2);
//		}
//		/* footer */
////		a0 = _mm_set_epi64x(mzd_read_bits(A, i, 64, acolhi), mzd_read_bits(A, i, 0, 64));
//		a0 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct?
//		if (i+1 < arows)
//		{
////			a1 = _mm_set_epi64x(mzd_read_bits(A, i+1, 64, acolhi), mzd_read_bits(A, i+1, 0, 64));
//			a1 = _mm_loadu_si128((__m128i*)A->rows[i+1]); // FIXME: always correct?
//		}
//		else
//		{
//			a1 = _mm_setzero_si128();
//		}
//		rr2 = _mm256_set_m128i(a1, a0);
//		m2  = _mm256_and_si256(vv, m1);
//		m2  = _mm256_cmpeq_epi64(m1, m2);
//		if (i < vcolhi)
//		{
////			a0 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, acolhi), mzd_read_bits(A, i+64, 0, 64));
//			a0 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct?
//			if (i+65 < arows)
//			{
////				a1 = _mm_set_epi64x(mzd_read_bits(A, i+65, 64, acolhi), mzd_read_bits(A, i+65, 0, 64));
//				a1 = _mm_loadu_si128((__m128i*)A->rows[i+65]); // FIXME: always correct?
//			}
//			else
//			{
//				a1 = _mm_setzero_si128();
//			}
//			rr3 = _mm256_set_m128i(a1, a0);
//			m3  = _mm256_shuffle_epi32(m2,0xFF);
//			rr3 = _mm256_and_si256(rr3, m3);
//			acc = _mm256_xor_si256(acc, rr3);
//		}
//		m2  = _mm256_shuffle_epi32(m2,0x00);
//		rr2 = _mm256_and_si256(rr2, m2);
//		acc = _mm256_xor_si256(acc, rr2);
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, 64, 0);
//			mzd_and_bits(res, j, 64, acolhi, 0);
//		}
//		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 0));
//		mzd_xor_bits(res, j, 64, acolhi, _mm256_extract_epi64(acc, 1));
//		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 2));
//		mzd_xor_bits(res, j, 64, acolhi, _mm256_extract_epi64(acc, 3));
//	}
//
//	return;
//}
//
///* = 256 = */
//
///* An SSE implem could also be added... */
//
///* Assumes that V->ncols and A->ncols > 128, otherwise a better implementation is selected */
///* FIXME Warning: also makes strong (stronger than above) assumptions about the matrix
// * format, namely that even if only 3 words are necessary (<= 192), a last one of padding
// * will always be here and we can load two 128-bit words */
//void mul_va256256_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
////	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vcmid = 0;
//	unsigned vchi  = 0;
//	unsigned acmid = 0;
//	unsigned achi  = 0;
//
//	if (vcols > 192)
//	{
//		vchi  = vcols - 192;
//		vcmid = 64;
//	}
//	else
//	{
//		vcmid = vcols - 128;
//	}
//	if (acols > 192)
//	{
//		achi  = acols - 192;
//		acmid = 64;
//	}
//	else
//	{
//		acmid = acols - 128;
//	}
//
//	__m256i v;
//
////	assert(vcols <= 256);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
//
////		v   = _mm256_set_epi64x(mzd_read_bits(V, j, 192, vchi), mzd_read_bits(V, j, 128, vcmid), mzd_read_bits(V, j, 64, 64), mzd_read_bits(V, j, 0, 64));
//		v = _mm256_loadu_si256((__m256i*)V->rows[j]); // FIXME: always correct??
//
//		__m256i m2, m3, m4, m5, rr2, rr3, rr4, rr5;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set1_epi64x(1);
//
//		for (int i = 0; i < 64; i++)
//		{
////			rr2 = _mm256_set_epi64x(mzd_read_bits(A, i, 192, achi), mzd_read_bits(A, i, 128, acmid), mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
//			rr2 = _mm256_loadu_si256((__m256i*)A->rows[i]); // FIXME: always correct??
//			m2  = _mm256_and_si256(v, m1);
//			m2  = _mm256_cmpeq_epi64(m1, m2);
////			rr3 = _mm256_set_epi64x(mzd_read_bits(A, i+64, 192, achi), mzd_read_bits(A, i+64, 128, acmid), mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
//			rr3 = _mm256_loadu_si256((__m256i*)A->rows[i+64]); // FIXME: always correct??
//			m3  = _mm256_permute4x64_epi64(m2, 0x55);
//			rr3 = _mm256_and_si256(rr3, m3);
//			acc = _mm256_xor_si256(acc, rr3);
//			if (i < vcmid)
//			{
////				rr4 = _mm256_set_epi64x(mzd_read_bits(A, i+128, 192, achi), mzd_read_bits(A, i+128, 128, acmid), mzd_read_bits(A, i+128, 64, 64), mzd_read_bits(A, i+128, 0, 64));
//				rr4 = _mm256_loadu_si256((__m256i*)A->rows[i+128]); // FIXME: always correct??
//				m4  = _mm256_permute4x64_epi64(m2, 0xAA);
//				rr4 = _mm256_and_si256(rr4, m4);
//				acc = _mm256_xor_si256(acc, rr4);
//			}
//			if (i < vchi)
//			{
////				rr5 = _mm256_set_epi64x(mzd_read_bits(A, i+192, 192, achi), mzd_read_bits(A, i+192, 128, acmid), mzd_read_bits(A, i+192, 64, 64), mzd_read_bits(A, i+192, 0, 64));
//				rr5 = _mm256_loadu_si256((__m256i*)A->rows[i+192]); // FIXME: always correct??
//				m5  = _mm256_permute4x64_epi64(m2, 0xFF);
//				rr5 = _mm256_and_si256(rr5, m5);
//				acc = _mm256_xor_si256(acc, rr5);
//			}
//			m2  = _mm256_permute4x64_epi64(m2, 0x00);
//			rr2 = _mm256_and_si256(rr2, m2);
//			acc = _mm256_xor_si256(acc, rr2);
//			m1  = _mm256_slli_epi64(m1, 1);
//		}
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, 64, 0);
//			mzd_and_bits(res, j, 64, 64, 0);
//			mzd_and_bits(res, j, 128, acmid, 0);
//			if (achi)
//				mzd_and_bits(res, j, 192, achi, 0);
//		}
//		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 0));
//		mzd_xor_bits(res, j, 64, 64, _mm256_extract_epi64(acc, 1));
//		mzd_xor_bits(res, j, 128, acmid, _mm256_extract_epi64(acc, 2));
//		if (achi)
//			mzd_xor_bits(res, j, 192, achi, _mm256_extract_epi64(acc, 3));
//	}
//
//	return;
//}
//
///*
// * Example of specialized function, as a model for future ones
// * assert(V->ncols > 224)
// * assert(V->ncols <= 256)
// * assert(A->ncols > 224)
// * assert(A->ncols <= 256)
// */
//void mul_224_224_bro_sse2(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vc192 = vcols - 192;
//	unsigned ac192 = acols - 192;
//
//	__m128i vlo, vhi;
//
//	for (int j = 0; j < vrows; j++)
//	{
//		__m128i acclo = _mm_setzero_si128();
//		__m128i acchi = _mm_setzero_si128();
//		__m128i ones  = _mm_set1_epi32(1);
//		__m128i m0, m00, m1, m2, m3, rl0, rh0, rl1, rh1, rl2, rh2, rl3, rh3;
//
////		vlo = _mm_set_epi64x(mzd_read_bits(V, j, 64, 64), mzd_read_bits(V, j, 0, 64));
//		vlo = _mm_loadu_si128((__m128i*)V->rows[j]); // FIXME: always correct??
////		vhi = _mm_set_epi64x(mzd_read_bits(V, j, 192, vc192), mzd_read_bits(V, j, 128, 64));
//		vhi = _mm_loadu_si128(((__m128i*)V->rows[j])+1); // FIXME: always correct??
//
//		for (int i = 0; i < 32; i++)
//		{
//			/* 128 Lo V columns */
//			m0 = _mm_and_si128(vlo, ones);
//			m0 = _mm_cmpeq_epi32(ones, m0);
//
//			m00 = _mm_shuffle_epi32(m0, 0x00);
////			rl0 = _mm_set_epi64x(mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
////			rh0 = _mm_set_epi64x(mzd_read_bits(A, i, 192, ac192), mzd_read_bits(A, i, 128, 64));
//			rl0 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct??
//			rh0 = _mm_loadu_si128(((__m128i*)A->rows[i])+1); // FIXME: always correct??
//			rl0 = _mm_and_si128(rl0, m00);
//			rh0 = _mm_and_si128(rh0, m00);
//
//			m1 = _mm_shuffle_epi32(m0, 0x55);
////			rl1 = _mm_set_epi64x(mzd_read_bits(A, i+32, 64, 64), mzd_read_bits(A, i+32, 0, 64));
////			rh1 = _mm_set_epi64x(mzd_read_bits(A, i+32, 192, ac192), mzd_read_bits(A, i+32, 128, 64));
//			rl1 = _mm_loadu_si128((__m128i*)A->rows[i+32]); // FIXME: always correct??
//			rh1 = _mm_loadu_si128(((__m128i*)A->rows[i+32])+1); // FIXME: always correct??
//			rl1 = _mm_and_si128(rl1, m1);
//			rh1 = _mm_and_si128(rh1, m1);
//
//			m2 = _mm_shuffle_epi32(m0, 0xAA);
////			rl2 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
////			rh2 = _mm_set_epi64x(mzd_read_bits(A, i+64, 192, ac192), mzd_read_bits(A, i+64, 128, 64));
//			rl2 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct??
//			rh2 = _mm_loadu_si128(((__m128i*)A->rows[i+64])+1); // FIXME: always correct??
//			rl2 = _mm_and_si128(rl2, m2);
//			rh2 = _mm_and_si128(rh2, m2);
//
//			m3 = _mm_shuffle_epi32(m0, 0xFF);
////			rl3 = _mm_set_epi64x(mzd_read_bits(A, i+96, 64, 64), mzd_read_bits(A, i+96, 0, 64));
////			rh3 = _mm_set_epi64x(mzd_read_bits(A, i+96, 192, ac192), mzd_read_bits(A, i+96, 128, 64));
//			rl3 = _mm_loadu_si128((__m128i*)A->rows[i+96]); // FIXME: always correct??
//			rh3 = _mm_loadu_si128(((__m128i*)A->rows[i+96])+1); // FIXME: always correct??
//			rl3 = _mm_and_si128(rl3, m3);
//			rh3 = _mm_and_si128(rh3, m3);
//
//			rl1 = _mm_xor_si128(rl1, rl2);
//			rh1 = _mm_xor_si128(rh1, rh2);
//			rl0 = _mm_xor_si128(rl0, rl3);
//			rh0 = _mm_xor_si128(rh0, rh3);
//			acclo = _mm_xor_si128(rl0, acclo);
//			acchi = _mm_xor_si128(rh0, acchi);
//			acclo = _mm_xor_si128(rl1, acclo);
//			acchi = _mm_xor_si128(rh1, acchi);
//
//			/* 128 Hi V columns */
//			m0 = _mm_and_si128(vhi, ones);
//			m0 = _mm_cmpeq_epi32(ones, m0);
//
//			m00 = _mm_shuffle_epi32(m0, 0x00);
////			rl0 = _mm_set_epi64x(mzd_read_bits(A, i+128, 64, 64), mzd_read_bits(A, i+128, 0, 64));
////			rh0 = _mm_set_epi64x(mzd_read_bits(A, i+128, 192, ac192), mzd_read_bits(A, i+128, 128, 64));
//			rl0 = _mm_loadu_si128((__m128i*)A->rows[i+128]); // FIXME: always correct??
//			rh0 = _mm_loadu_si128(((__m128i*)A->rows[i+128])+1); // FIXME: always correct??
//			rl0 = _mm_and_si128(rl0, m00);
//			rh0 = _mm_and_si128(rh0, m00);
//
//			m1 = _mm_shuffle_epi32(m0, 0x55);
////			rl1 = _mm_set_epi64x(mzd_read_bits(A, i+160, 64, 64), mzd_read_bits(A, i+160, 0, 64));
////			rh1 = _mm_set_epi64x(mzd_read_bits(A, i+160, 192, ac192), mzd_read_bits(A, i+160, 128, 64));
//			rl1 = _mm_loadu_si128((__m128i*)A->rows[i+160]); // FIXME: always correct??
//			rh1 = _mm_loadu_si128(((__m128i*)A->rows[i+160])+1); // FIXME: always correct??
//			rl1 = _mm_and_si128(rl1, m1);
//			rh1 = _mm_and_si128(rh1, m1);
//
//			m2 = _mm_shuffle_epi32(m0, 0xAA);
////			rl2 = _mm_set_epi64x(mzd_read_bits(A, i+192, 64, 64), mzd_read_bits(A, i+192, 0, 64));
////			rh2 = _mm_set_epi64x(mzd_read_bits(A, i+192, 192, ac192), mzd_read_bits(A, i+192, 128, 64));
//			rl2 = _mm_loadu_si128((__m128i*)A->rows[i+192]); // FIXME: always correct??
//			rh2 = _mm_loadu_si128(((__m128i*)A->rows[i+192])+1); // FIXME: always correct??
//			rl2 = _mm_and_si128(rl2, m2);
//			rh2 = _mm_and_si128(rh2, m2);
//
//			if (i+224 < vcols) // vcols for arows
//			{
//				m3 = _mm_shuffle_epi32(m0, 0xFF);
////				rl3 = _mm_set_epi64x(mzd_read_bits(A, i+224, 64, 64), mzd_read_bits(A, i+224, 0, 64));
////				rh3 = _mm_set_epi64x(mzd_read_bits(A, i+224, 192, ac192), mzd_read_bits(A, i+224, 128, 64));
//				rl3 = _mm_loadu_si128((__m128i*)A->rows[i+224]); // FIXME: always correct??
//				rh3 = _mm_loadu_si128(((__m128i*)A->rows[i+224])+1); // FIXME: always correct??
//				rl3 = _mm_and_si128(rl3, m3);
//				rh3 = _mm_and_si128(rh3, m3);
//				acclo = _mm_xor_si128(rl3, acclo);
//				acchi = _mm_xor_si128(rh3, acchi);
//			}
//
//			rl1 = _mm_xor_si128(rl1, rl2);
//			rh1 = _mm_xor_si128(rh1, rh2);
//			acclo = _mm_xor_si128(rl0, acclo);
//			acchi = _mm_xor_si128(rh0, acchi);
//			acclo = _mm_xor_si128(rl1, acclo);
//			acchi = _mm_xor_si128(rh1, acchi);
//
//			ones  = _mm_slli_epi32(ones, 1);
//		}
//
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, 64, 0);
//			mzd_and_bits(res, j, 64, 64, 0);
//			mzd_and_bits(res, j, 128, 64, 0);
//			mzd_and_bits(res, j, 192, ac192, 0);
//		}
////		mzd_xor_bits(res, j, 0, 64, _mm_extract_epi64(acclo, 0));
////		mzd_xor_bits(res, j, 64, 64, _mm_extract_epi64(acclo, 1));
////		mzd_xor_bits(res, j, 128, 64, _mm_extract_epi64(acchi, 0));
////		mzd_xor_bits(res, j, 192, ac192, _mm_extract_epi64(acchi, 1));
//		// for full SSE2 compatibility; only a moderate slowdown from _mm_extract_epi_64
//		mzd_xor_bits(res, j, 0, 16, _mm_extract_epi16(acclo, 0));
//		mzd_xor_bits(res, j, 16, 16, _mm_extract_epi16(acclo, 1));
//		mzd_xor_bits(res, j, 32, 16, _mm_extract_epi16(acclo, 2));
//		mzd_xor_bits(res, j, 48, 16, _mm_extract_epi16(acclo, 3));
//		mzd_xor_bits(res, j, 64, 16, _mm_extract_epi16(acclo, 4));
//		mzd_xor_bits(res, j, 80, 16, _mm_extract_epi16(acclo, 5));
//		mzd_xor_bits(res, j, 96, 16, _mm_extract_epi16(acclo, 6));
//		mzd_xor_bits(res, j, 112, 16, _mm_extract_epi16(acclo, 7));
//		mzd_xor_bits(res, j, 128, 16, _mm_extract_epi16(acchi, 0));
//		mzd_xor_bits(res, j, 144, 16, _mm_extract_epi16(acchi, 1));
//		mzd_xor_bits(res, j, 160, 16, _mm_extract_epi16(acchi, 2));
//		mzd_xor_bits(res, j, 176, 16, _mm_extract_epi16(acchi, 3));
//		mzd_xor_bits(res, j, 192, 16, _mm_extract_epi16(acchi, 4)); // FIXME: assume leftover is always zero... correct?
//		mzd_xor_bits(res, j, 208, 16, _mm_extract_epi16(acchi, 5)); // ditto
//		mzd_xor_bits(res, j, 224, 16, _mm_extract_epi16(acchi, 6)); // ditto
//		mzd_xor_bits(res, j, 240, 16, _mm_extract_epi16(acchi, 7)); // ditto
//	}
//
//	return;
//}

/*
 * TESTS
 */

/* = 32 = */


//void test_speed_32(unsigned iter)
//{
//	struct timeval tv1, tv2;
//	uint64_t tusec;
//	mzd_t *x, *y, *a;
//
//	x = mzd_init(32, 32);
//	y = mzd_init(32, 32);
//	a = mzd_init(32, 32);
//
//	mzd_randomize_custom(x, &my_little_rand, NULL);
//	mzd_randomize_custom(a, &my_little_rand, NULL);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va3232_avx(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 32 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va3232_sse(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 32 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va3232_std(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 32 w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va6464_std(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 32(64) w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mzd_mul_m4rm(y, x, a, 0);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	return;
//}
//
///* = 64 = */
//
//void test_speed_64(unsigned iter)
//{
//	struct timeval tv1, tv2;
//	uint64_t tusec;
//	mzd_t *x, *y, *a;
//
//	x = mzd_init(64, 64);
//	y = mzd_init(64, 64);
//	a = mzd_init(64, 64);
//
//	mzd_randomize_custom(x, &my_little_rand, NULL);
//	mzd_randomize_custom(a, &my_little_rand, NULL);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va6464_avx(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 64 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va6464_sse(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 64 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va6464_std(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 64 w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mzd_mul_m4rm(y, x, a, 0);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	return;
//}
//
///* = 128 */
//
//void test_correc_full_128(int tries)
//{
//	mzd_t *x, *a;
//	mzd_t *ym4r, *ystd, *ysse, *yavx;
//
//	for (int c1 = 65; c1 <= 128; c1++)
//	{
//		for (int r = 1; r <= 1; r++)
//		{
//			for (int c2 = 65; c2 <= 128; c2++)
//			{
//				x = mzd_init(r, c1);
//				a = mzd_init(c1, c2);
//				ym4r = mzd_init(r, c2);
//				ystd = mzd_init(r, c2);
//				ysse = mzd_init(r, c2);
//				yavx = mzd_init(r, c2);
//
//				int avxg = 1;
//				int sseg = 1;
//				int stdg = 1;
//
//				for (int i = 0; i < tries; i++)
//				{
//					mzd_randomize_custom(x, &my_little_rand, NULL);
//					mzd_randomize_custom(a, &my_little_rand, NULL);
//
//					mul_va128128_avx(yavx, x, a, 1);
//					mul_va128128_sse(ysse, x, a, 1);
//					mul_va128128_std(ystd, x, a, 1);
//					mzd_mul_m4rm(ym4r, x, a, 0);
//
//					avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
//					sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
//					stdg = mzd_cmp(ystd, ym4r) == 0 ? stdg : 0;
//				}
//				if (!stdg)
//					printf("[%dx%d X %dx%d (#%d)] STD: BAD\n", r, c1, c1, c2, tries);
//				if (!sseg)
//					printf("[%dx%d X %dx%d (#%d)] SSE: BAD\n", r, c1, c1, c2, tries);
//				if (!avxg)
//					printf("[%dx%d X %dx%d (#%d)] AVX: BAD\n", r, c1, c1, c2, tries);
//			}
//		}
//	}
//	printf("Full 128... done\n");
//
//	return;
//}
//
//void test_speed_128(unsigned iter)
//{
//	struct timeval tv1, tv2;
//	uint64_t tusec;
//	mzd_t *x, *y, *a;
//
//	x = mzd_init(128, 128);
//	y = mzd_init(128, 128);
//	a = mzd_init(128, 128);
//
//	mzd_randomize_custom(x, &my_little_rand, NULL);
//	mzd_randomize_custom(a, &my_little_rand, NULL);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va128128_avx(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 128 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va128128_sse(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 128 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va128128_std(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 128 w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mzd_mul_m4rm(y, x, a, 0);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	return;
//}
//
///* = 256 */
//
//void test_correc_full_256(int tries)
//{
//	mzd_t *x, *a;
//	mzd_t *ym4r, *yavx, *ysse;
//
//	for (int c1 = 225; c1 <= 256; c1++)
//	{
//		for (int r = 1; r <= 1; r++)
//		{
//			for (int c2 = 225; c2 <= 256; c2++)
//			{
//				x = mzd_init(r, c1);
//				a = mzd_init(c1, c2);
//				ym4r = mzd_init(r, c2);
//				yavx = mzd_init(r, c2);
//				ysse = mzd_init(r, c2);
//
//				int avxg = 1;
//				int sseg = 1;
//
//				for (int i = 0; i < tries; i++)
//				{
//					mzd_randomize_custom(x, &my_little_rand, NULL);
//					mzd_randomize_custom(a, &my_little_rand, NULL);
//
//					mul_va256256_avx(yavx, x, a, 1);
//					mul_224_224_bro_sse2(ysse, x, a, 1);
//					mzd_mul_m4rm(ym4r, x, a, 0);
//
//					avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
//					sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
//				}
//				if (!avxg)
//					printf("[%dx%d X %dx%d (#%d)] AVX: BAD\n", r, c1, c1, c2, tries);
//				if (!sseg)
//					printf("[%dx%d X %dx%d (#%d)] SSE: BAD\n", r, c1, c1, c2, tries);
//			}
//		}
//	}
//	printf("Full 256... done\n");
//
//	return;
//}
//
//void test_speed_256(unsigned iter)
//{
//	struct timeval tv1, tv2;
//	uint64_t tusec;
//	mzd_t *x, *y, *a;
//
//	x = mzd_init(256, 256);
//	y = mzd_init(256, 256);
//	a = mzd_init(256, 256);
//
//	mzd_randomize_custom(x, &my_little_rand, NULL);
//	mzd_randomize_custom(a, &my_little_rand, NULL);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_va256256_avx(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 256 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		mul_224_224_bro_sse2(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("`Fast' 256 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
////		_mzd_mul_va(y, x, a, 1);
//		mzd_mul_m4rm(y, x, a, 0);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
//
//	return;
//}


void test_correc_bro_ska(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *yska; 

	for (int c1 = 1; c1 <= 128; c1++)
	{
		for (int r = 1; r <= 1; r++)
		{
			for (int c2 = 1; c2 <= 128; c2++)
			{
				x = mzd_init(r, c1);
				a = mzd_init(c1, c2);
				ym4r = mzd_init(r, c2);
				yska = mzd_init(r, c2);

				int skag = 1;

				for (int i = 0; i < tries; i++)
				{
					mzd_randomize_custom(x, &my_little_rand, NULL);
					mzd_randomize_custom(a, &my_little_rand, NULL);

					mul_bro_ska(yska, x, a, 1);
					mzd_mul_m4rm(ym4r, x, a, 0);

					skag = mzd_cmp(yska, ym4r) == 0 ? skag : 0;
				}
				if (!skag)
					printf("[%dx%d X %dx%d (#%d)] SKA: BAD\n", r, c1, c1, c2, tries);
			}
		}
	}
	printf("SKA correctness test... done\n");

	return;
}

void test_perf_bro_ska(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;
	
	for (int c1 = 16; c1 <= 128; c1 += 16)
	{
		for (int r = 16; r <= 128; r += 32)
		{
			for (int c2 = 16; c2 <= 128; c2 += 16)
			{
				x = mzd_init(r, c1);
				a = mzd_init(c1, c2);
				y = mzd_init(r, c2);

				mzd_randomize_custom(x, &my_little_rand, NULL);
				mzd_randomize_custom(a, &my_little_rand, NULL);

				gettimeofday(&tv1, NULL);
				for (unsigned i = 0; i < iter; i++)
					mul_bro_ska(y, x, a, 1);
				gettimeofday(&tv2, NULL);
				tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
				printf("BRO SKA (%dx%d X %dx%d):\t %llu usecs (#%u) [%f usecs/op]\n", r, c1, c1, c2, tusec, iter, (double)tusec / (double)iter);

				gettimeofday(&tv1, NULL);
				for (unsigned i = 0; i < iter; i++)
					mzd_mul_m4rm(y, x, a, 0);
				gettimeofday(&tv2, NULL);
				tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
				printf("M4RI (%dx%d X %dx%d):\t\t %llu usecs (#%u) [%f usecs/op]\n", r, c1, c1, c2, tusec, iter, (double)tusec / (double)iter);
			}
		}
	}

	return;
}


int main()
{
//	test_correc_bro_ska(1 << 8);
	test_perf_bro_ska(1 << 16);

	return 0;
}

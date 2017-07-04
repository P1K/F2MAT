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

/* Kept here pour mémoire; will be refactored in different files (bro_ska, bro_sse2, bro_avx2) */

/*
 * As a general rule, performance is much better when the dimensions are known
 * at compilation time (as it speeds up the matrix read & write operation
 * considerably)
 * FIXME: most of the speed can be gained back by issuing direct memory loads,
 * but will this be always correct?
 * => Actually, the main point is to ensure that (some of the) unitialized data is zero-filled
 * => But even if it's not, it's not a problem: one just copies the meaningful part of the result at the end
 * => So everything should be allright in the end
 * (the funny masks like
	uint64_t vcolmask = vcols == 128 ? ~0ull : (1ull << (vcols - 64)) - 1ull;
	are not even necessary, it's not faster than standard mzd r/w (which I actually find surprising!))
 */


/*
 * For custom randomization thru callback
 */
uint64_t my_little_rand(void *willignore)
{
	return (((uint64_t)(hc128random()) << 32) ^ ((uint64_t)hc128random()));
}

// printf("%016llX%016llX\n", _mm_extract_epi64(rr, 0), _mm_extract_epi64(rr, 1));

/* = 32 = */

/*
 * All input allocated
 * if clear, reset res before adding the result
 * Computes VxA, where A is at most 32x32
 */
/* Actually not useful (on 64-bit archs); 6464_std can be used instead */
void mul_va3232_std(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
//	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
//	assert(vcols <= 32);
//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{
		uint32_t m, acc = 0;
//		uint32_t v = mzd_read_bits(V, j, 0, vcols);
		uint32_t v = *(V->rows[j]);//mzd_read_bits(V, j, 0, vcols);

		v = ~v;
		for (int i = 0; i < vcols; i++)
		{
			m = (v & 1) - 1;
//			acc ^= (m & mzd_read_bits(A, i, 0, acols));
			acc ^= (m & *(A->rows[i]));
			v >>= 1;
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		mzd_xor_bits(res, j, 0, acols, acc);
	}

	return;
}

/*
 * Same as above, but using SSE4.1
 * (lower version could be used in a rather similar way)
 */
void mul_va3232_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
//	assert(vcols <= 32);
//	assert(arows == vcols);


	for (int j = 0; j < vrows; j++)
	{
//		uint64_t v = mzd_read_bits(V, j, 0, vcols);
		uint64_t v = *(V->rows[j]); // FIXME: always correct?

		__m128i m2, rr;
		__m128i acc = _mm_setzero_si128();
		__m128i m1  = _mm_set_epi32(1, 2, 4, 8);
		__m128i vv  = _mm_set1_epi32(v);

		int i;
		/* main loop */
		for (i = 0; i < (int)(vcols-4); i+=4)
		{
//			rr  = _mm_set_epi32(mzd_read_bits(A, i, 0, acols), mzd_read_bits(A, i+1, 0, acols), mzd_read_bits(A, i+2, 0, acols), mzd_read_bits(A, i+3, 0, acols));
			rr  = _mm_set_epi32(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3])); // FIXME: always correct?
			m2  = _mm_and_si128(vv, m1);
			m2  = _mm_cmpeq_epi32(m1, m2);
			rr  = _mm_and_si128(rr, m2);
			acc = _mm_xor_si128(acc, rr);
			m1  = _mm_slli_epi32(m1, 4);
		}
		/* footer */
//		uint32_t r0 = mzd_read_bits(A, i, 0, acols);
//		uint32_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
//		uint32_t r2 = i+2 >= arows ? 0 : mzd_read_bits(A, i+2, 0, acols);
//		uint32_t r3 = i+3 >= arows ? 0 : mzd_read_bits(A, i+3, 0, acols);
		uint32_t r0 = *(A->rows[i]); // FIXME: always correct?
		uint32_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME: always correct?
		uint32_t r2 = i+2 >= arows ? 0 : *(A->rows[i+2]); // FIXME: always correct?
		uint32_t r3 = i+3 >= arows ? 0 : *(A->rows[i+3]); // FIXME: always correct?
		rr  = _mm_set_epi32(r0, r1, r2, r3);
		m2  = _mm_and_si128(vv, m1);
		m2  = _mm_cmpeq_epi32(m1, m2);
		rr  = _mm_and_si128(rr, m2);
		acc = _mm_xor_si128(acc, rr);

		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		uint32_t tmpacc1 = _mm_extract_epi32(acc, 0) ^ _mm_extract_epi32(acc, 1);
		uint32_t tmpacc2 = _mm_extract_epi32(acc, 2) ^ _mm_extract_epi32(acc, 3);
		tmpacc1 ^= tmpacc2;
		mzd_xor_bits(res, j, 0, acols, tmpacc1);
	}

	return;
}

/*
 * Same as above, but using AVX2
 */
void mul_va3232_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
//	assert(vcols <= 32);
//	assert(arows == vcols);


	for (int j = 0; j < vrows; j++)
	{
//		uint64_t v = mzd_read_bits(V, j, 0, vcols);
		uint64_t v = *(V->rows[j]); // FIXME: always correct?

		__m256i m2, rr;
		__m256i acc = _mm256_setzero_si256();
		__m256i m1  = _mm256_set_epi32(1, 2, 4, 8, 16, 32, 64, 128);
		__m256i vv  = _mm256_set1_epi32(v);

		/* main loop */
		int i;
		for (i = 0; i < (int)(vcols-8); i+=8)
		{
//			rr  = _mm256_set_epi32(mzd_read_bits(A, i, 0, acols), mzd_read_bits(A, i+1, 0, acols), mzd_read_bits(A, i+2, 0, acols), mzd_read_bits(A, i+3, 0, acols),
//					mzd_read_bits(A, i+4, 0, acols), mzd_read_bits(A, i+5, 0, acols), mzd_read_bits(A, i+6, 0, acols), mzd_read_bits(A, i+7, 0, acols));
			rr  = _mm256_set_epi32(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3]), *(A->rows[i+4]), *(A->rows[i+5]), *(A->rows[i+6]), *(A->rows[i+7])); // FIXME: always correct?
			m2  = _mm256_and_si256(vv, m1);
			m2  = _mm256_cmpeq_epi32(m1, m2);
			rr  = _mm256_and_si256(rr, m2);
			acc = _mm256_xor_si256(acc, rr);
			m1  = _mm256_slli_epi32(m1, 8);
		}
		/* footer */
//		uint32_t r0 = mzd_read_bits(A, i, 0, acols);
//		uint32_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
//		uint32_t r2 = i+2 >= arows ? 0 : mzd_read_bits(A, i+2, 0, acols);
//		uint32_t r3 = i+3 >= arows ? 0 : mzd_read_bits(A, i+3, 0, acols);
//		uint32_t r4 = i+4 >= arows ? 0 : mzd_read_bits(A, i+4, 0, acols);
//		uint32_t r5 = i+5 >= arows ? 0 : mzd_read_bits(A, i+5, 0, acols);
//		uint32_t r6 = i+6 >= arows ? 0 : mzd_read_bits(A, i+6, 0, acols);
//		uint32_t r7 = i+7 >= arows ? 0 : mzd_read_bits(A, i+7, 0, acols);
		uint32_t r0 = *(A->rows[i]); // FIXME: always correct?
		uint32_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME: always correct?
		uint32_t r2 = i+2 >= arows ? 0 : *(A->rows[i+2]); // FIXME: always correct?
		uint32_t r3 = i+3 >= arows ? 0 : *(A->rows[i+3]); // FIXME: always correct?
		uint32_t r4 = i+4 >= arows ? 0 : *(A->rows[i+4]); // FIXME: always correct?
		uint32_t r5 = i+5 >= arows ? 0 : *(A->rows[i+5]); // FIXME: always correct?
		uint32_t r6 = i+6 >= arows ? 0 : *(A->rows[i+6]); // FIXME: always correct?
		uint32_t r7 = i+7 >= arows ? 0 : *(A->rows[i+7]); // FIXME: always correct?
		rr  = _mm256_set_epi32(r0, r1, r2, r3, r4, r5, r6, r7);
		m2  = _mm256_and_si256(vv, m1);
		m2  = _mm256_cmpeq_epi32(m1, m2);
		rr  = _mm256_and_si256(rr, m2);
		acc = _mm256_xor_si256(acc, rr);

		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		uint32_t tmpacc1 = _mm256_extract_epi32(acc, 0) ^ _mm256_extract_epi32(acc, 1);
		uint32_t tmpacc2 = _mm256_extract_epi32(acc, 2) ^ _mm256_extract_epi32(acc, 3);
		uint32_t tmpacc3 = _mm256_extract_epi32(acc, 4) ^ _mm256_extract_epi32(acc, 5);
		uint32_t tmpacc4 = _mm256_extract_epi32(acc, 6) ^ _mm256_extract_epi32(acc, 7);
		tmpacc1 ^= tmpacc2;
		tmpacc3 ^= tmpacc4;
		tmpacc1 ^= tmpacc3;
		mzd_xor_bits(res, j, 0, acols, tmpacc1);
	}

	return;
}

/* = 64 */

/*
 * All input allocated
 * if clear, reset res before adding the result
 * Computes VxA, where A is at most 64x64
 */
void mul_va6464_std(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
//	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
//	assert(vcols <= 64);
//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{
		uint64_t m, acc = 0;
//		uint64_t v = mzd_read_bits(V, j, 0, vcols);
		uint64_t v = *(V->rows[j]); // FIXME: always correct?

		v = ~v;
		for (int i = 0; i < vcols; i++)
		{
			m = (v & 1) - 1;
//			acc ^= (m & mzd_read_bits(A, i, 0, acols));
			acc ^= (m & *(A->rows[i])); // FIXME: always correct?
			v >>= 1;
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		mzd_xor_bits(res, j, 0, acols, acc);
	}

	return;
}

/*
 * Same as above, but using SSE4.1
 * (lower version could be used in a rather similar way)
 */
void mul_va6464_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
//	assert(vcols <= 64);
//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{
//		uint64_t v = mzd_read_bits(V, j, 0, vcols);
		uint64_t v = *(V->rows[j]); // FIXME: always correct?

		__m128i m2, rr;
		__m128i acc = _mm_setzero_si128();
		__m128i m1  = _mm_set_epi64x(1, 2);
		__m128i vv  = _mm_set1_epi64x(v);

		int i;
		/* main loop */
		for (i = 0; i < (int)(vcols-2); i+=2)
		{
//			rr  = _mm_set_epi64x(mzd_read_bits(A, i, 0, acols), mzd_read_bits(A, i+1, 0, acols));
			rr  = _mm_set_epi64x(*(A->rows[i]),*(A->rows[i+1])); // FIXME: always correct?
			m2  = _mm_and_si128(vv, m1);
			m2  = _mm_cmpeq_epi64(m1, m2);
			rr  = _mm_and_si128(rr, m2);
			acc = _mm_xor_si128(acc, rr);
			m1  = _mm_slli_epi64(m1, 2);
		}
		/* footer */
//		uint64_t r0 = mzd_read_bits(A, i, 0, acols);
//		uint64_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
		uint64_t r0 = *(A->rows[i]); // FIXME: always correct?
		uint64_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME : always correct?
		rr  = _mm_set_epi64x(r0, r1);
//		rr = _mm_loadu_si128((__m128i*)A->rows[i]); // unfortunately, the matrix structure doesn't allow to do this much faster load
//		(not suitable for generic form anyway)
		m2  = _mm_and_si128(vv, m1);
		m2  = _mm_cmpeq_epi64(m1, m2);
		rr  = _mm_and_si128(rr, m2);
		acc = _mm_xor_si128(acc, rr);

		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		mzd_xor_bits(res, j, 0, acols, _mm_extract_epi64(acc, 0));
		mzd_xor_bits(res, j, 0, acols, _mm_extract_epi64(acc, 1));
	}

	return;
}

/*
 * Same as above, but using AVX2
 */
void mul_va6464_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
	//	assert(vcols <= 64);
	//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{
//		uint64_t v = mzd_read_bits(V, j, 0, vcols);
		uint64_t v = *(V->rows[j]); // FIXME: always correct?

		__m256i m2, rr;
		__m256i acc = _mm256_setzero_si256();
		__m256i m1  = _mm256_set_epi64x(1, 2, 4, 8);
		__m256i vv  = _mm256_set1_epi64x(v);

		int i;
		/* main loop */
		for (i = 0; i < (int)(vcols-4); i+=4)
		{
//			rr  = _mm256_set_epi64x(mzd_read_bits(A, i, 0, 64), mzd_read_bits(A, i+1, 0, 64), mzd_read_bits(A, i+2, 0, 64), mzd_read_bits(A, i+3, 0, 64));
			//		rr  = _mm256_loadu_si256((__m256i*)(A->rows[i])); // unfortunately, the matrix structure doesn't allow to do this much faster load
			rr  = _mm256_set_epi64x(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3])); // FIXME: always correct?
			//		ditto
			m2  = _mm256_and_si256(vv, m1);
			m2  = _mm256_cmpeq_epi64(m1, m2);
			rr  = _mm256_and_si256(rr, m2);
			acc = _mm256_xor_si256(acc, rr);
			m1  = _mm256_slli_epi64(m1, 4);
		}
		/* footer */
//		uint64_t r0 = mzd_read_bits(A, i, 0, acols);
//		uint64_t r1 = i+1 >= arows ? 0 : mzd_read_bits(A, i+1, 0, acols);
//		uint64_t r2 = i+2 >= arows ? 0 : mzd_read_bits(A, i+2, 0, acols);
//		uint64_t r3 = i+3 >= arows ? 0 : mzd_read_bits(A, i+3, 0, acols);
		uint64_t r0 = *(A->rows[i]); // FIXME: always correct?
		uint64_t r1 = i+1 >= arows ? 0 : *(A->rows[i+1]); // FIXME; always correct?
		uint64_t r2 = i+2 >= arows ? 0 : *(A->rows[i+2]); // FIXME; always correct;
		uint64_t r3 = i+3 >= arows ? 0 : *(A->rows[i+3]); // FIXME; always correct;
		rr  = _mm256_set_epi64x(r0, r1, r2, r3);
		m2  = _mm256_and_si256(vv, m1);
		m2  = _mm256_cmpeq_epi64(m1, m2);
		rr  = _mm256_and_si256(rr, m2);
		acc = _mm256_xor_si256(acc, rr);

		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		uint64_t tmpacc1 = _mm256_extract_epi64(acc, 0) ^ _mm256_extract_epi64(acc, 1);
		uint64_t tmpacc2 = _mm256_extract_epi64(acc, 2) ^ _mm256_extract_epi64(acc, 3);
		tmpacc1 ^= tmpacc2;
		mzd_xor_bits(res, j, 0, acols, tmpacc1);
	}

	return;
}

/* = 128 = */

/*
 * All input allocated
 * if clear, reset res before adding the result
 * Computes VxA, where A is at most 128x128
 * Warning: also needs V to have > 64 columns (otherwise, a smaller implementation is selected)
 * Warning: also needs A to have > 64 columns (otherwise, a smaller implementation is selected)
 */
void mul_va128128_std(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
//	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
//	uint64_t vcolmask = vcols == 128 ? ~0ull : (1ull << (vcols - 64)) - 1ull;
//	uint64_t acolmask = acols == 128 ? ~0ull : (1ull << (acols - 64)) - 1ull;
	
	for (int j = 0; j < vrows; j++)
	{
		uint64_t mlo, mhi;
		uint64_t acc0lo = 0;
		uint64_t acc0hi = 0;
		uint64_t acc1lo = 0;
		uint64_t acc1hi = 0;
		uint64_t vlo = mzd_read_bits(V, j, 0, 64);
		uint64_t vhi = mzd_read_bits(V, j, 64, vcols-64);
//		uint64_t vhi = mzd_read_bits(V, j, 64, 64) & vcolmask;
//		uint64_t vlo = *(V->rows[j]); // FIXME: always correct?
//		uint64_t vhi = *(V->rows[j]+1); // FIXME: always correct?

		vlo = ~vlo;
		vhi = ~vhi;
		for (int i = 0; i < 64; i++)
		{
			mlo = (vlo & 1) - 1;
			mhi = (vhi & 1) - 1;
			acc0lo ^= (mlo & mzd_read_bits(A, i, 0, 64));
//			acc0hi ^= (mlo & mzd_read_bits(A, i, 64, acols-64));
			acc0hi ^= (mlo & mzd_read_bits(A, i, 64, 64)); // leftover is simply not copied
//			acc0lo ^= (mlo & *(A->rows[i])); // FIXME: always correct?
//			acc0hi ^= (mlo & *(A->rows[i]+1)); // FIXME: always correct?
			if (i+64 < vcols)
			{
				acc1lo ^= (mhi & mzd_read_bits(A, i+64, 0, 64));
//				acc1hi ^= (mhi & mzd_read_bits(A, i+64, 64, acols-64));
				acc1hi ^= (mhi & mzd_read_bits(A, i+64, 64, 64)); // ditto
//				acc1lo ^= (mhi & *(A->rows[i+64])); // FIXME: always correct?
//				acc1hi ^= (mhi & *(A->rows[i+64]+1)); // FIXME: always correct?
			}
			vlo >>= 1;
			vhi >>= 1;
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
//			mzd_and_bits(res, j, 64, acols-64, 0);
			mzd_and_bits(res, j, 64, 64, 0);
		}
		mzd_xor_bits(res, j, 0, 64, acc0lo);
		mzd_xor_bits(res, j, 0, 64, acc1lo);
		mzd_xor_bits(res, j, 64, acols - 64, acc0hi);
		mzd_xor_bits(res, j, 64, acols - 64, acc1hi);
	}

	return;
}

/*
 * All input allocated
 * if clear, reset res before adding the result
 * Computes VxA, where A is at most 128x128
 * Warning: also needs V to have > 64 columns (otherwise, a smaller implementation is selected)
 * Warning: also needs A to have > 64 columns (otherwise, a smaller implementation is selected)
 */
void mul_va128128_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
//	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
	unsigned vcolhi = vcols - 64;
	unsigned acolhi = acols - 64;
//	assert(vcols <= 128);
//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{
//		__m128i v  = _mm_set_epi64x(mzd_read_bits(V, j, 64, vcolhi), mzd_read_bits(V, j, 0, 64));
		__m128i v  = _mm_loadu_si128((__m128i*)V->rows[j]);

		__m128i m2, m3, rr2, rr3;
		__m128i acc = _mm_setzero_si128();
		__m128i m1  = _mm_set1_epi64x(1);

		for (int i = 0; i < 64; i++)
		{
//			rr2  = _mm_set_epi64x(mzd_read_bits(A, i, 64, acolhi), mzd_read_bits(A, i, 0, 64));
			rr2 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct?
			m2  = _mm_and_si128(v, m1);
			m2  = _mm_cmpeq_epi64(m1, m2);
			if (i < vcolhi)
			{
//				rr3 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, acolhi), mzd_read_bits(A, i+64, 0, 64));
				rr3 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct?
				m3  = _mm_shuffle_epi32(m2,0xFF);
				rr3 = _mm_and_si128(rr3, m3);
				acc = _mm_xor_si128(acc, rr3);
			}
			m2  = _mm_shuffle_epi32(m2,0x00);
			rr2 = _mm_and_si128(rr2, m2);
			acc = _mm_xor_si128(acc, rr2);
			m1  = _mm_slli_epi64(m1, 1);
		}

		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
			mzd_and_bits(res, j, 64, acolhi, 0);
		}
		mzd_xor_bits(res, j, 0, 64, _mm_extract_epi64(acc, 0));
		mzd_xor_bits(res, j, 64, acolhi, _mm_extract_epi64(acc, 1));
	}

	return;
}
// fully general: more complex and slower
//void mul_va128128_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
////	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vcollo = vcols >= 64 ? 64 : vcols;
//	unsigned vcolhi = vcols > 64 ? vcols - 64 : 0;
//	unsigned acollo = acols >= 64 ? 64 : acols;
//	unsigned acolhi = acols > 64 ? acols - 64 : 0;
////	assert(vcols <= 128);
////	assert(arows == vcols);
//
////	acols = 128;
////	vcols = 128;
////	vrows = 1;
////	vcollo = 64;
////	vcolhi = 64;
////	acollo = 64;
////	acolhi = 64;
//	for (int j = 0; j < vrows; j++)
//	{
//		uint64_t v01 = vcolhi ? mzd_read_bits(V, j, 64, vcolhi) : 0;
//
//		__m128i v  = _mm_set_epi64x(v01, mzd_read_bits(V, j, 0, vcollo));
//
//		__m128i m2, m3, rr2, rr3;
//		__m128i acc = _mm_setzero_si128();
//		__m128i m1  = _mm_set1_epi64x(1);
//
//		for (int i = 0; i < vcollo; i++)
//		{
//			uint64_t a01 = acolhi ? mzd_read_bits(A, i, 64, acolhi) : 0;
//
//			rr2  = _mm_set_epi64x(a01, mzd_read_bits(A, i, 0, acollo));
//			m2  = _mm_and_si128(v, m1);
//			m2  = _mm_cmpeq_epi64(m1, m2);
//			if (i < vcolhi)
//			{
//				a01 = acolhi ? mzd_read_bits(A, i+64, 64, acolhi) : 0;
//
//				rr3 = _mm_set_epi64x(a01, mzd_read_bits(A, i+64, 0, acollo));
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
//			mzd_and_bits(res, j, 0, acollo, 0);
//			if (acolhi)
//				mzd_and_bits(res, j, 64, acolhi, 0);
//		}
//		mzd_xor_bits(res, j, 0, acollo, _mm_extract_epi64(acc, 0));
//		if (acolhi)
//			mzd_xor_bits(res, j, 64, acolhi, _mm_extract_epi64(acc, 1));
//	}
//
//	return;
//}
// Fixed 128 vcols version, quite simpler
//void mul_va128128_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	__m128i v  = _mm_set_epi64x(mzd_read_bits(V, 0, 64, 64), mzd_read_bits(V, 0, 0, 64));
//
//	__m128i m2, m3, rr2, rr3;
//	__m128i acc = _mm_setzero_si128();
//	__m128i m1  = _mm_set1_epi64x(1);
//
//	for (int i = 0; i < 64; i++)
//	{
//		rr2  = _mm_set_epi64x(mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
//		rr3  = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
////		rr2 = _mm_loadu_si128((__m128i*)A->rows[i]); // now fine but not really faster
////		rr3 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // now fine but not really faster
//		m2  = _mm_and_si128(v, m1);
//		// there's a tradeoff between comp granularity and code/masks size
//		m2  = _mm_cmpeq_epi64(m1, m2);
//		m3  = _mm_shuffle_epi32(m2,0xFF);
//		m2  = _mm_shuffle_epi32(m2,0x00);
//		rr2 = _mm_and_si128(rr2, m2);
//		rr3 = _mm_and_si128(rr3, m3);
//		acc = _mm_xor_si128(acc, rr2);
//		acc = _mm_xor_si128(acc, rr3);
//		m1  = _mm_slli_epi64(m1, 1);
//	}
//
//	if (clear)
//	{
//		mzd_and_bits(res, 0, 0, 64, 0);
//		mzd_and_bits(res, 0, 64, 64, 0);
//	}
//	mzd_xor_bits(res, 0, 0, 64, _mm_extract_epi64(acc, 0));
//	mzd_xor_bits(res, 0, 64, 64, _mm_extract_epi64(acc, 1));
//
//	return;
//}

/* same not generic as above */
void mul_va128128_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
	unsigned vcolhi = vcols - 64;
	unsigned acolhi = acols - 64;
//	assert(vcols <= 128);
//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{
//		__m128i v  = _mm_set_epi64x(mzd_read_bits(V, j, 64, vcolhi), mzd_read_bits(V, j, 0, 64));
		__m128i v  = _mm_loadu_si128((__m128i*)V->rows[j]); // FIXME: always correct?

		__m256i m2, m3, rr2, rr3;
		__m256i acc = _mm256_setzero_si256();
		__m256i m1  = _mm256_set_epi64x(2, 2, 1, 1);
		__m256i vv  = _mm256_set_m128i(v, v);
		__m128i a0;
		__m128i a1;

		int i;
		/* main loop */
		for (i = 0; i < 62; i+=2)
		{
//			a0 = _mm_set_epi64x(mzd_read_bits(A, i, 64, acolhi), mzd_read_bits(A, i, 0, 64));
//			a1 = _mm_set_epi64x(mzd_read_bits(A, i+1, 64, acolhi), mzd_read_bits(A, i+1, 0, 64));
			a0 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct?
			a1 = _mm_loadu_si128((__m128i*)A->rows[i+1]); // FIXME: always correct?

			rr2 = _mm256_set_m128i(a1, a0);
			m2  = _mm256_and_si256(vv, m1);
			m2  = _mm256_cmpeq_epi64(m1, m2);
			if (i < vcolhi)
			{
//				a0 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, acolhi), mzd_read_bits(A, i+64, 0, 64));
				a0 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct?
				if (i+65 < arows)
				{
//					a1 = _mm_set_epi64x(mzd_read_bits(A, i+65, 64, acolhi), mzd_read_bits(A, i+65, 0, 64));
					a1 = _mm_loadu_si128((__m128i*)A->rows[i+65]); // FIXME: always correct?
				}
				else
				{
					a1 = _mm_setzero_si128();
				}
				rr3 = _mm256_set_m128i(a1, a0);
				m3  = _mm256_shuffle_epi32(m2,0xFF);
				rr3 = _mm256_and_si256(rr3, m3);
				acc = _mm256_xor_si256(acc, rr3);
			}
			m2  = _mm256_shuffle_epi32(m2,0x00);
			rr2 = _mm256_and_si256(rr2, m2);
			acc = _mm256_xor_si256(acc, rr2);
			m1  = _mm256_slli_epi64(m1, 2);
		}
		/* footer */
//		a0 = _mm_set_epi64x(mzd_read_bits(A, i, 64, acolhi), mzd_read_bits(A, i, 0, 64));
		a0 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct?
		if (i+1 < arows)
		{
//			a1 = _mm_set_epi64x(mzd_read_bits(A, i+1, 64, acolhi), mzd_read_bits(A, i+1, 0, 64));
			a1 = _mm_loadu_si128((__m128i*)A->rows[i+1]); // FIXME: always correct?
		}
		else
		{
			a1 = _mm_setzero_si128();
		}
		rr2 = _mm256_set_m128i(a1, a0);
		m2  = _mm256_and_si256(vv, m1);
		m2  = _mm256_cmpeq_epi64(m1, m2);
		if (i < vcolhi)
		{
//			a0 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, acolhi), mzd_read_bits(A, i+64, 0, 64));
			a0 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct?
			if (i+65 < arows)
			{
//				a1 = _mm_set_epi64x(mzd_read_bits(A, i+65, 64, acolhi), mzd_read_bits(A, i+65, 0, 64));
				a1 = _mm_loadu_si128((__m128i*)A->rows[i+65]); // FIXME: always correct?
			}
			else
			{
				a1 = _mm_setzero_si128();
			}
			rr3 = _mm256_set_m128i(a1, a0);
			m3  = _mm256_shuffle_epi32(m2,0xFF);
			rr3 = _mm256_and_si256(rr3, m3);
			acc = _mm256_xor_si256(acc, rr3);
		}
		m2  = _mm256_shuffle_epi32(m2,0x00);
		rr2 = _mm256_and_si256(rr2, m2);
		acc = _mm256_xor_si256(acc, rr2);

		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
			mzd_and_bits(res, j, 64, acolhi, 0);
		}
		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 0));
		mzd_xor_bits(res, j, 64, acolhi, _mm256_extract_epi64(acc, 1));
		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 2));
		mzd_xor_bits(res, j, 64, acolhi, _mm256_extract_epi64(acc, 3));
	}

	return;
}
// generic, slow
//void mul_va128128_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vcollo = vcols >= 64 ? 64 : vcols;
//	unsigned vcolhi = vcols > 64 ? vcols - 64 : 0;
//	unsigned acollo = acols >= 64 ? 64 : acols;
//	unsigned acolhi = acols > 64 ? acols - 64 : 0;
////	assert(vcols <= 128);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
//		uint64_t v01 = vcolhi ? mzd_read_bits(V, j, 64, vcolhi) : 0;
//
//		__m128i v  = _mm_set_epi64x(v01, mzd_read_bits(V, j, 0, vcollo));
//
//		__m256i m2, m3, rr2, rr3;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set_epi64x(2, 2, 1, 1);
//		__m256i vv  = _mm256_set_m128i(v, v);
//
//		int i;
//		/* main loop */
//		for (i = 0; i < ((int)vcollo)-2; i+=2)
//		{
//			uint64_t a01 = acolhi ? mzd_read_bits(A, i, 64, acolhi) : 0;
//			uint64_t a11 = acolhi ? mzd_read_bits(A, i+1, 64, acolhi) : 0;
//
//			rr2 = _mm256_set_epi64x(a11, mzd_read_bits(A, i+1, 0, acollo), a01, mzd_read_bits(A, i, 0, acollo));
//			m2  = _mm256_and_si256(vv, m1);
//			m2  = _mm256_cmpeq_epi64(m1, m2);
//			if (i < vcolhi)
//			{
//				a01 = acolhi ? mzd_read_bits(A, i+64, 64, acolhi) : 0;
//				uint64_t r3lo = 0;
//				uint64_t r3hi = 0;
//				if (i+65 < arows)
//				{
//					r3lo = mzd_read_bits(A, i+65, 0, acollo);
//					r3hi = acolhi ? mzd_read_bits(A, i+65, 64, acolhi) : 0;
//				}
//				rr3 = _mm256_set_epi64x(r3hi, r3lo, a01, mzd_read_bits(A, i+64, 0, acollo));
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
//		uint64_t r0lo = mzd_read_bits(A, i, 0, acollo);
//		uint64_t r0hi = acolhi ? mzd_read_bits(A, i, 64, acolhi) : 0;
//		uint64_t r1lo = 0;
//		uint64_t r1hi = 0;
//		if (i+1 < arows)
//		{
//			r1lo = mzd_read_bits(A, i+1, 0, acollo);
//			r1hi = acolhi ? mzd_read_bits(A, i+1, 64, acolhi) : 0;
//		}
//		rr2 = _mm256_set_epi64x(r1hi, r1lo, r0hi, r0lo);
//		m2  = _mm256_and_si256(vv, m1);
//		m2  = _mm256_cmpeq_epi64(m1, m2);
//		if (i < vcolhi)
//		{
//			uint64_t r2lo = mzd_read_bits(A, i+64, 0, acollo);
//			uint64_t r2hi = acolhi ? mzd_read_bits(A, i+64, 64, acolhi) : 0;
//			uint64_t r3lo = 0;
//			uint64_t r3hi = 0;
//			if (i+65 < arows)
//			{
//				r3lo = mzd_read_bits(A, i+65, 0, acollo);
//				r3hi = acolhi ? mzd_read_bits(A, i+65, 64, acolhi) : 0;
//			}
//			rr3 = _mm256_set_epi64x(r3hi, r3lo, r2hi, r2lo);
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
//			mzd_and_bits(res, j, 0, acollo, 0);
//			mzd_and_bits(res, j, 64, acolhi, 0);
//		}
//		mzd_xor_bits(res, j, 0, acollo, _mm256_extract_epi64(acc, 0));
//		mzd_xor_bits(res, j, 64, acolhi, _mm256_extract_epi64(acc, 1));
//		mzd_xor_bits(res, j, 0, acollo, _mm256_extract_epi64(acc, 2));
//		mzd_xor_bits(res, j, 64, acolhi, _mm256_extract_epi64(acc, 3));
//	}
//
//	return;
//}
// fixed dim, fast
//void mul_va128128_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	__m128i v  = _mm_set_epi64x(mzd_read_bits(V, 0, 64, 64), mzd_read_bits(V, 0, 0, 64));
//
//	__m256i m2, m3, rr2, rr3;
//	__m256i acc = _mm256_setzero_si256();
//	__m256i m1  = _mm256_set_epi64x(2, 2, 1, 1);
//	__m256i vv  = _mm256_set_m128i(v, v);
//
//	for (int i = 0; i < 64; i+=2)
//	{
////		rr2 = _mm256_set_epi64x(mzd_read_bits(A, i+1, 64, 64), mzd_read_bits(A, i+1, 0, 64), mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
////		rr3 = _mm256_set_epi64x(mzd_read_bits(A, i+65, 64, 64), mzd_read_bits(A, i+65, 0, 64), mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
//		rr2 = _mm256_loadu_si256((__m256i*)A->rows[i]); // makes an assumption on the matrix representation, but quite faster
//		rr3 = _mm256_loadu_si256((__m256i*)A->rows[i+64]); // ditto
//		m2  = _mm256_and_si256(vv, m1);
//		m2  = _mm256_cmpeq_epi64(m1, m2);
//		m3  = _mm256_shuffle_epi32(m2,0xFF);
//		m2  = _mm256_shuffle_epi32(m2,0x00);
//		rr2 = _mm256_and_si256(rr2, m2);
//		rr3 = _mm256_and_si256(rr3, m3);
//		acc = _mm256_xor_si256(acc, rr2);
//		acc = _mm256_xor_si256(acc, rr3);
//		m1  = _mm256_slli_epi64(m1, 2);
//	}
//	if (clear)
//	{
//		mzd_and_bits(res, 0, 0, 64, 0);
//		mzd_and_bits(res, 0, 64, 64, 0);
//	}
//	mzd_xor_bits(res, 0, 0, 64, _mm256_extract_epi64(acc, 0));
//	mzd_xor_bits(res, 0, 64, 64, _mm256_extract_epi64(acc, 1));
//	mzd_xor_bits(res, 0, 0, 64, _mm256_extract_epi64(acc, 2));
//	mzd_xor_bits(res, 0, 64, 64, _mm256_extract_epi64(acc, 3));
//
//	return;
//}

/* = 256 = */

/* An SSE implem could also be added... */

/* Assumes that V->ncols and A->ncols > 128, otherwise a better implementation is selected */
/* FIXME Warning: also makes strong (stronger than above) assumptions about the matrix
 * format, namely that even if only 3 words are necessary (<= 192), a last one of padding
 * will always be here and we can load two 128-bit words */
void mul_va256256_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
//	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
	unsigned vcmid = 0;
	unsigned vchi  = 0;
	unsigned acmid = 0;
	unsigned achi  = 0;

	if (vcols > 192)
	{
		vchi  = vcols - 192;
		vcmid = 64;
	}
	else
	{
		vcmid = vcols - 128;
	}
	if (acols > 192)
	{
		achi  = acols - 192;
		acmid = 64;
	}
	else
	{
		acmid = acols - 128;
	}

	__m256i v;

//	assert(vcols <= 256);
//	assert(arows == vcols);

	for (int j = 0; j < vrows; j++)
	{

//		v   = _mm256_set_epi64x(mzd_read_bits(V, j, 192, vchi), mzd_read_bits(V, j, 128, vcmid), mzd_read_bits(V, j, 64, 64), mzd_read_bits(V, j, 0, 64));
		v = _mm256_loadu_si256((__m256i*)V->rows[j]); // FIXME: always correct??

		__m256i m2, m3, m4, m5, rr2, rr3, rr4, rr5;
		__m256i acc = _mm256_setzero_si256();
		__m256i m1  = _mm256_set1_epi64x(1);

		for (int i = 0; i < 64; i++)
		{
//			rr2 = _mm256_set_epi64x(mzd_read_bits(A, i, 192, achi), mzd_read_bits(A, i, 128, acmid), mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
			rr2 = _mm256_loadu_si256((__m256i*)A->rows[i]); // FIXME: always correct??
			m2  = _mm256_and_si256(v, m1);
			m2  = _mm256_cmpeq_epi64(m1, m2);
//			rr3 = _mm256_set_epi64x(mzd_read_bits(A, i+64, 192, achi), mzd_read_bits(A, i+64, 128, acmid), mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
			rr3 = _mm256_loadu_si256((__m256i*)A->rows[i+64]); // FIXME: always correct??
			m3  = _mm256_permute4x64_epi64(m2, 0x55);
			rr3 = _mm256_and_si256(rr3, m3);
			acc = _mm256_xor_si256(acc, rr3);
			if (i < vcmid)
			{
//				rr4 = _mm256_set_epi64x(mzd_read_bits(A, i+128, 192, achi), mzd_read_bits(A, i+128, 128, acmid), mzd_read_bits(A, i+128, 64, 64), mzd_read_bits(A, i+128, 0, 64));
				rr4 = _mm256_loadu_si256((__m256i*)A->rows[i+128]); // FIXME: always correct??
				m4  = _mm256_permute4x64_epi64(m2, 0xAA);
				rr4 = _mm256_and_si256(rr4, m4);
				acc = _mm256_xor_si256(acc, rr4);
			}
			if (i < vchi)
			{
//				rr5 = _mm256_set_epi64x(mzd_read_bits(A, i+192, 192, achi), mzd_read_bits(A, i+192, 128, acmid), mzd_read_bits(A, i+192, 64, 64), mzd_read_bits(A, i+192, 0, 64));
				rr5 = _mm256_loadu_si256((__m256i*)A->rows[i+192]); // FIXME: always correct??
				m5  = _mm256_permute4x64_epi64(m2, 0xFF);
				rr5 = _mm256_and_si256(rr5, m5);
				acc = _mm256_xor_si256(acc, rr5);
			}
			m2  = _mm256_permute4x64_epi64(m2, 0x00);
			rr2 = _mm256_and_si256(rr2, m2);
			acc = _mm256_xor_si256(acc, rr2);
			m1  = _mm256_slli_epi64(m1, 1);
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
			mzd_and_bits(res, j, 64, 64, 0);
			mzd_and_bits(res, j, 128, acmid, 0);
			if (achi)
				mzd_and_bits(res, j, 192, achi, 0);
		}
		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 0));
		mzd_xor_bits(res, j, 64, 64, _mm256_extract_epi64(acc, 1));
		mzd_xor_bits(res, j, 128, acmid, _mm256_extract_epi64(acc, 2));
		if (achi)
			mzd_xor_bits(res, j, 192, achi, _mm256_extract_epi64(acc, 3));
	}

	return;
}

// generic; complex and slow
//void mul_va256256_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
////	unsigned arows = A->nrows;
//	unsigned acols = A->ncols;
//	unsigned vcols = V->ncols;
//	unsigned vrows = V->nrows;
//	unsigned vcol00, vcol01, vcol02, vcol03;
//	unsigned acol00, acol01, acol02, acol03;
//
//	vcol00 = 0;
//	vcol01 = 0;
//	vcol02 = 0;
//	vcol03 = 0;
//	acol00 = 0;
//	acol01 = 0;
//	acol02 = 0;
//	acol03 = 0;
//	if (vcols > 192)
//	{
//		vcol03 = vcols - 192;
//		vcol02 = 64;
//		vcol01 = 64;
//		vcol00 = 64;
//	}
//	else if (vcols > 128)
//	{
//		vcol02 = vcols - 128;
//		vcol01 = 64;
//		vcol00 = 64;
//	}
//	else if (vcols > 64)
//	{
//		vcol01 = vcols - 64;
//		vcol00 = 64;
//	}
//	else
//	{
//		vcol00 = vcols;
//	}
//	if (acols > 192)
//	{
//		acol03 = acols - 192;
//		acol02 = 64;
//		acol01 = 64;
//		acol00 = 64;
//	}
//	else if (acols > 128)
//	{
//		acol02 = acols - 128;
//		acol01 = 64;
//		acol00 = 64;
//	}
//	else if (acols > 64)
//	{
//		acol01 = acols - 64;
//		acol00 = 64;
//	}
//	else
//	{
//		acol00 = acols;
//	}
//
////	assert(vcols <= 256);
////	assert(arows == vcols);
//
//	for (int j = 0; j < vrows; j++)
//	{
//		uint64_t v01, v02, v03;
//
//		v03 = vcol03 ? mzd_read_bits(V, j, 192, vcol03) : 0;
//		v02 = vcol02 ? mzd_read_bits(V, j, 128, vcol02) : 0;
//		v01 = vcol01 ? mzd_read_bits(V, j, 64, vcol01) : 0;
//		__m256i v  = _mm256_set_epi64x(v03, v02, v01, mzd_read_bits(V, j, 0, vcol00));
//
//		__m256i m2, m3, m4, m5, rr2, rr3, rr4, rr5;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set1_epi64x(1);
//
//		for (int i = 0; i < vcol00; i++)
//		{
//			uint64_t a01, a02, a03;
//
//			a03 = acol03 ? mzd_read_bits(A, i, 192, acol03) : 0;
//			a02 = acol02 ? mzd_read_bits(A, i, 128, acol02) : 0;
//			a01 = acol01 ? mzd_read_bits(A, i, 64, acol01) : 0;
//
//			rr2 = _mm256_set_epi64x(a03, a02, a01, mzd_read_bits(A, i, 0, acol00));
//			m2  = _mm256_and_si256(v, m1);
//			m2  = _mm256_cmpeq_epi64(m1, m2);
//			if (i < vcol01)
//			{
//				a03 = acol03 ? mzd_read_bits(A, i+64, 192, acol03) : 0;
//				a02 = acol02 ? mzd_read_bits(A, i+64, 128, acol02) : 0;
//				a01 = acol01 ? mzd_read_bits(A, i+64, 64, acol01) : 0;
//
//				rr3 = _mm256_set_epi64x(a03, a02, a01, mzd_read_bits(A, i+64, 0, acol00));
//				m3  = _mm256_permute4x64_epi64(m2, 0x55);
//				rr3 = _mm256_and_si256(rr3, m3);
//				acc = _mm256_xor_si256(acc, rr3);
//			}
//			if (i < vcol02)
//			{
//				a03 = acol03 ? mzd_read_bits(A, i+128, 192, acol03) : 0;
//				a02 = acol02 ? mzd_read_bits(A, i+128, 128, acol02) : 0;
//				a01 = acol01 ? mzd_read_bits(A, i+128, 64, acol01) : 0;
//
//				rr4 = _mm256_set_epi64x(a03, a02, a01, mzd_read_bits(A, i+128, 0, acol00));
//				m4  = _mm256_permute4x64_epi64(m2, 0xAA);
//				rr4 = _mm256_and_si256(rr4, m4);
//				acc = _mm256_xor_si256(acc, rr4);
//			}
//			if (i < vcol03)
//			{
//				a03 = acol03 ? mzd_read_bits(A, i+192, 192, acol03) : 0;
//				a02 = acol02 ? mzd_read_bits(A, i+192, 128, acol02) : 0;
//				a01 = acol01 ? mzd_read_bits(A, i+192, 64, acol01) : 0;
//
//				rr5 = _mm256_set_epi64x(a03, a02, a01, mzd_read_bits(A, i+192, 0, acol00));
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
//			mzd_and_bits(res, j, 0, acol00, 0);
//			if (acol01)
//				mzd_and_bits(res, j, 64, acol01, 0);
//			if (acol02)
//				mzd_and_bits(res, j, 128, acol02, 0);
//			if (acol03)
//				mzd_and_bits(res, j, 192, acol03, 0);
//		}
//		mzd_xor_bits(res, j, 0, acol00, _mm256_extract_epi64(acc, 0));
//		if (acol01)
//			mzd_xor_bits(res, j, 64, acol01, _mm256_extract_epi64(acc, 1));
//		if (acol02)
//			mzd_xor_bits(res, j, 128, acol02, _mm256_extract_epi64(acc, 2));
//		if (acol03)
//			mzd_xor_bits(res, j, 192, acol03, _mm256_extract_epi64(acc, 3));
//	}
//
//	return;
//}
// fixed dim, faster
//void mul_va256256_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
//{
//	for (int j = 0; j < 256; j++)
//	{
//		__m256i v  = _mm256_set_epi64x(mzd_read_bits(V, j, 192, 64), mzd_read_bits(V, j, 128, 64), mzd_read_bits(V, j, 64, 64), mzd_read_bits(V, j, 0, 64));
//
//		__m256i m2, m3, m4, m5, rr2, rr3, rr4, rr5;
//		__m256i acc = _mm256_setzero_si256();
//		__m256i m1  = _mm256_set1_epi64x(1);
//
//		for (int i = 0; i < 64; i++)
//		{
//			rr2 = _mm256_set_epi64x(mzd_read_bits(A, i, 192, 64), mzd_read_bits(A, i, 128, 64), mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
//			rr3 = _mm256_set_epi64x(mzd_read_bits(A, i+64, 192, 64), mzd_read_bits(A, i+64, 128, 64), mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
//			rr4 = _mm256_set_epi64x(mzd_read_bits(A, i+128, 192, 64), mzd_read_bits(A, i+128, 128, 64), mzd_read_bits(A, i+128, 64, 64), mzd_read_bits(A, i+128, 0, 64));
//			rr5 = _mm256_set_epi64x(mzd_read_bits(A, i+192, 192, 64), mzd_read_bits(A, i+192, 128, 64), mzd_read_bits(A, i+192, 64, 64), mzd_read_bits(A, i+192, 0, 64));
//			//		rr2 = _mm256_loadu_si256((__m256i*)A->rows[i]); // now fine but not really faster
//			//		rr3 = _mm256_loadu_si256((__m256i*)A->rows[i+64]);
//			//		rr4 = _mm256_loadu_si256((__m256i*)A->rows[i+128]);
//			//		rr5 = _mm256_loadu_si256((__m256i*)A->rows[i+192]);
//			m2  = _mm256_and_si256(v, m1);
//			// ditto, tradeoff
//			m2  = _mm256_cmpeq_epi64(m1, m2);
//			m5  = _mm256_permute4x64_epi64(m2, 0xFF);
//			m4  = _mm256_permute4x64_epi64(m2, 0xAA);
//			m3  = _mm256_permute4x64_epi64(m2, 0x55);
//			m2  = _mm256_permute4x64_epi64(m2, 0x00);
//			rr2 = _mm256_and_si256(rr2, m2);
//			rr3 = _mm256_and_si256(rr3, m3);
//			rr4 = _mm256_and_si256(rr4, m4);
//			rr5 = _mm256_and_si256(rr5, m5);
//			rr2 = _mm256_xor_si256(rr3, rr2);
//			rr4 = _mm256_xor_si256(rr5, rr4);
//			acc = _mm256_xor_si256(acc, rr2);
//			acc = _mm256_xor_si256(acc, rr4);
//			m1  = _mm256_slli_epi64(m1, 1);
//		}
//		if (clear)
//		{
//			mzd_and_bits(res, j, 0, 64, 0);
//			mzd_and_bits(res, j, 64, 64, 0);
//			mzd_and_bits(res, j, 128, 64, 0);
//			mzd_and_bits(res, j, 192, 64, 0);
//		}
//		mzd_xor_bits(res, j, 0, 64, _mm256_extract_epi64(acc, 0));
//		mzd_xor_bits(res, j, 64, 64, _mm256_extract_epi64(acc, 1));
//		mzd_xor_bits(res, j, 128, 64, _mm256_extract_epi64(acc, 2));
//		mzd_xor_bits(res, j, 192, 64, _mm256_extract_epi64(acc, 3));
//	}
//
//	return;
//}

/*
 * Example of specialized function, as a model for future ones
 * assert(V->ncols > 224)
 * assert(V->ncols <= 256)
 * assert(A->ncols > 224)
 * assert(A->ncols <= 256)
 */
void mul_224_224_bro_sse2(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;
	unsigned vc192 = vcols - 192;
	unsigned ac192 = acols - 192;

	__m128i vlo, vhi;

	for (int j = 0; j < vrows; j++)
	{
		__m128i acclo = _mm_setzero_si128();
		__m128i acchi = _mm_setzero_si128();
		__m128i ones  = _mm_set1_epi32(1);
		__m128i m0, m00, m1, m2, m3, rl0, rh0, rl1, rh1, rl2, rh2, rl3, rh3;

//		vlo = _mm_set_epi64x(mzd_read_bits(V, j, 64, 64), mzd_read_bits(V, j, 0, 64));
		vlo = _mm_loadu_si128((__m128i*)V->rows[j]); // FIXME: always correct??
//		vhi = _mm_set_epi64x(mzd_read_bits(V, j, 192, vc192), mzd_read_bits(V, j, 128, 64));
		vhi = _mm_loadu_si128(((__m128i*)V->rows[j])+1); // FIXME: always correct??

		for (int i = 0; i < 32; i++)
		{
			/* 128 Lo V columns */
			m0 = _mm_and_si128(vlo, ones);
			m0 = _mm_cmpeq_epi32(ones, m0);

			m00 = _mm_shuffle_epi32(m0, 0x00);
//			rl0 = _mm_set_epi64x(mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
//			rh0 = _mm_set_epi64x(mzd_read_bits(A, i, 192, ac192), mzd_read_bits(A, i, 128, 64));
			rl0 = _mm_loadu_si128((__m128i*)A->rows[i]); // FIXME: always correct??
			rh0 = _mm_loadu_si128(((__m128i*)A->rows[i])+1); // FIXME: always correct??
			rl0 = _mm_and_si128(rl0, m00);
			rh0 = _mm_and_si128(rh0, m00);

			m1 = _mm_shuffle_epi32(m0, 0x55);
//			rl1 = _mm_set_epi64x(mzd_read_bits(A, i+32, 64, 64), mzd_read_bits(A, i+32, 0, 64));
//			rh1 = _mm_set_epi64x(mzd_read_bits(A, i+32, 192, ac192), mzd_read_bits(A, i+32, 128, 64));
			rl1 = _mm_loadu_si128((__m128i*)A->rows[i+32]); // FIXME: always correct??
			rh1 = _mm_loadu_si128(((__m128i*)A->rows[i+32])+1); // FIXME: always correct??
			rl1 = _mm_and_si128(rl1, m1);
			rh1 = _mm_and_si128(rh1, m1);

			m2 = _mm_shuffle_epi32(m0, 0xAA);
//			rl2 = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
//			rh2 = _mm_set_epi64x(mzd_read_bits(A, i+64, 192, ac192), mzd_read_bits(A, i+64, 128, 64));
			rl2 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // FIXME: always correct??
			rh2 = _mm_loadu_si128(((__m128i*)A->rows[i+64])+1); // FIXME: always correct??
			rl2 = _mm_and_si128(rl2, m2);
			rh2 = _mm_and_si128(rh2, m2);

			m3 = _mm_shuffle_epi32(m0, 0xFF);
//			rl3 = _mm_set_epi64x(mzd_read_bits(A, i+96, 64, 64), mzd_read_bits(A, i+96, 0, 64));
//			rh3 = _mm_set_epi64x(mzd_read_bits(A, i+96, 192, ac192), mzd_read_bits(A, i+96, 128, 64));
			rl3 = _mm_loadu_si128((__m128i*)A->rows[i+96]); // FIXME: always correct??
			rh3 = _mm_loadu_si128(((__m128i*)A->rows[i+96])+1); // FIXME: always correct??
			rl3 = _mm_and_si128(rl3, m3);
			rh3 = _mm_and_si128(rh3, m3);

			rl1 = _mm_xor_si128(rl1, rl2);
			rh1 = _mm_xor_si128(rh1, rh2);
			rl0 = _mm_xor_si128(rl0, rl3);
			rh0 = _mm_xor_si128(rh0, rh3);
			acclo = _mm_xor_si128(rl0, acclo);
			acchi = _mm_xor_si128(rh0, acchi);
			acclo = _mm_xor_si128(rl1, acclo);
			acchi = _mm_xor_si128(rh1, acchi);

			/* 128 Hi V columns */
			m0 = _mm_and_si128(vhi, ones);
			m0 = _mm_cmpeq_epi32(ones, m0);

			m00 = _mm_shuffle_epi32(m0, 0x00);
//			rl0 = _mm_set_epi64x(mzd_read_bits(A, i+128, 64, 64), mzd_read_bits(A, i+128, 0, 64));
//			rh0 = _mm_set_epi64x(mzd_read_bits(A, i+128, 192, ac192), mzd_read_bits(A, i+128, 128, 64));
			rl0 = _mm_loadu_si128((__m128i*)A->rows[i+128]); // FIXME: always correct??
			rh0 = _mm_loadu_si128(((__m128i*)A->rows[i+128])+1); // FIXME: always correct??
			rl0 = _mm_and_si128(rl0, m00);
			rh0 = _mm_and_si128(rh0, m00);

			m1 = _mm_shuffle_epi32(m0, 0x55);
//			rl1 = _mm_set_epi64x(mzd_read_bits(A, i+160, 64, 64), mzd_read_bits(A, i+160, 0, 64));
//			rh1 = _mm_set_epi64x(mzd_read_bits(A, i+160, 192, ac192), mzd_read_bits(A, i+160, 128, 64));
			rl1 = _mm_loadu_si128((__m128i*)A->rows[i+160]); // FIXME: always correct??
			rh1 = _mm_loadu_si128(((__m128i*)A->rows[i+160])+1); // FIXME: always correct??
			rl1 = _mm_and_si128(rl1, m1);
			rh1 = _mm_and_si128(rh1, m1);

			m2 = _mm_shuffle_epi32(m0, 0xAA);
//			rl2 = _mm_set_epi64x(mzd_read_bits(A, i+192, 64, 64), mzd_read_bits(A, i+192, 0, 64));
//			rh2 = _mm_set_epi64x(mzd_read_bits(A, i+192, 192, ac192), mzd_read_bits(A, i+192, 128, 64));
			rl2 = _mm_loadu_si128((__m128i*)A->rows[i+192]); // FIXME: always correct??
			rh2 = _mm_loadu_si128(((__m128i*)A->rows[i+192])+1); // FIXME: always correct??
			rl2 = _mm_and_si128(rl2, m2);
			rh2 = _mm_and_si128(rh2, m2);

			if (i+224 < vcols) // vcols for arows
			{
				m3 = _mm_shuffle_epi32(m0, 0xFF);
//				rl3 = _mm_set_epi64x(mzd_read_bits(A, i+224, 64, 64), mzd_read_bits(A, i+224, 0, 64));
//				rh3 = _mm_set_epi64x(mzd_read_bits(A, i+224, 192, ac192), mzd_read_bits(A, i+224, 128, 64));
				rl3 = _mm_loadu_si128((__m128i*)A->rows[i+224]); // FIXME: always correct??
				rh3 = _mm_loadu_si128(((__m128i*)A->rows[i+224])+1); // FIXME: always correct??
				rl3 = _mm_and_si128(rl3, m3);
				rh3 = _mm_and_si128(rh3, m3);
				acclo = _mm_xor_si128(rl3, acclo);
				acchi = _mm_xor_si128(rh3, acchi);
			}

			rl1 = _mm_xor_si128(rl1, rl2);
			rh1 = _mm_xor_si128(rh1, rh2);
			acclo = _mm_xor_si128(rl0, acclo);
			acchi = _mm_xor_si128(rh0, acchi);
			acclo = _mm_xor_si128(rl1, acclo);
			acchi = _mm_xor_si128(rh1, acchi);

			ones  = _mm_slli_epi32(ones, 1);
		}

		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
			mzd_and_bits(res, j, 64, 64, 0);
			mzd_and_bits(res, j, 128, 64, 0);
			mzd_and_bits(res, j, 192, ac192, 0);
		}
//		mzd_xor_bits(res, j, 0, 64, _mm_extract_epi64(acclo, 0));
//		mzd_xor_bits(res, j, 64, 64, _mm_extract_epi64(acclo, 1));
//		mzd_xor_bits(res, j, 128, 64, _mm_extract_epi64(acchi, 0));
//		mzd_xor_bits(res, j, 192, ac192, _mm_extract_epi64(acchi, 1));
		// for full SSE2 compatibility; only a moderate slowdown from _mm_extract_epi_64
		mzd_xor_bits(res, j, 0, 16, _mm_extract_epi16(acclo, 0));
		mzd_xor_bits(res, j, 16, 16, _mm_extract_epi16(acclo, 1));
		mzd_xor_bits(res, j, 32, 16, _mm_extract_epi16(acclo, 2));
		mzd_xor_bits(res, j, 48, 16, _mm_extract_epi16(acclo, 3));
		mzd_xor_bits(res, j, 64, 16, _mm_extract_epi16(acclo, 4));
		mzd_xor_bits(res, j, 80, 16, _mm_extract_epi16(acclo, 5));
		mzd_xor_bits(res, j, 96, 16, _mm_extract_epi16(acclo, 6));
		mzd_xor_bits(res, j, 112, 16, _mm_extract_epi16(acclo, 7));
		mzd_xor_bits(res, j, 128, 16, _mm_extract_epi16(acchi, 0));
		mzd_xor_bits(res, j, 144, 16, _mm_extract_epi16(acchi, 1));
		mzd_xor_bits(res, j, 160, 16, _mm_extract_epi16(acchi, 2));
		mzd_xor_bits(res, j, 176, 16, _mm_extract_epi16(acchi, 3));
		mzd_xor_bits(res, j, 192, 16, _mm_extract_epi16(acchi, 4)); // FIXME: assume leftover is always zero... correct?
		mzd_xor_bits(res, j, 208, 16, _mm_extract_epi16(acchi, 5)); // ditto
		mzd_xor_bits(res, j, 224, 16, _mm_extract_epi16(acchi, 6)); // ditto
		mzd_xor_bits(res, j, 240, 16, _mm_extract_epi16(acchi, 7)); // ditto
	}

	return;
}

/*
 * TESTS
 */

/* = 32 = */

void test_correc_matvec_32(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *ystd, *ysse, *yavx;

	x = mzd_init(1, 32);
	a = mzd_init(32, 32);
	ym4r = mzd_init(1, 32);
	ystd = mzd_init(1, 32);
	ysse = mzd_init(1, 32);
	yavx = mzd_init(1, 32);

	int avxg = 1;
	int sseg = 1;
	int stdg = 1;

	for (int i = 0; i < tries; i++)
	{
		mzd_randomize_custom(x, &my_little_rand, NULL);
		mzd_randomize_custom(a, &my_little_rand, NULL);

		mul_va3232_avx(yavx, x, a, 1);
		mul_va3232_sse(ysse, x, a, 1);
		mul_va3232_std(ystd, x, a, 1);
		mzd_mul_m4rm(ym4r, x, a, 0);

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
		sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
		stdg = mzd_cmp(ystd, ym4r) == 0 ? stdg : 0;
//		printf("%08llX\n", mzd_read_bits(ym4r, 0, 0, 32));
	}
	printf("[32 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");
	printf("[32 (#%d)] SSE: %s\n", tries, sseg ? "good" : "bad");
	printf("[32 (#%d)] STD: %s\n", tries, stdg ? "good" : "bad");

	return;
}

void test_correc_full_32(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *ystd, *ysse, *yavx;

	for (int c1 = 1; c1 <= 32; c1++)
	{
		for (int r = 1; r <= 1; r++)
		{
			for (int c2 = 1; c2 <= 32; c2++)
			{
				x = mzd_init(r, c1);
				a = mzd_init(c1, c2);
				ym4r = mzd_init(r, c2);
				ystd = mzd_init(r, c2);
				ysse = mzd_init(r, c2);
				yavx = mzd_init(r, c2);

				int avxg = 1;
				int sseg = 1;
				int stdg = 1;

				for (int i = 0; i < tries; i++)
				{
					mzd_randomize_custom(x, &my_little_rand, NULL);
					mzd_randomize_custom(a, &my_little_rand, NULL);

					mul_va3232_avx(yavx, x, a, 1);
					mul_va3232_sse(ysse, x, a, 1);
					mul_va3232_std(ystd, x, a, 1);
					mzd_mul_m4rm(ym4r, x, a, 0);

					avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
					sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
					stdg = mzd_cmp(ystd, ym4r) == 0 ? stdg : 0;
				}
				if (!stdg)
					printf("[%dx%d X %dx%d (#%d)] STD: BAD\n", r, c1, c1, c2, tries);
				if (!sseg)
					printf("[%dx%d X %dx%d (#%d)] SSE: BAD\n", r, c1, c1, c2, tries);
				if (!avxg)
					printf("[%dx%d X %dx%d (#%d)] AVX: BAD\n", r, c1, c1, c2, tries);
			}
		}
	}
	printf("Full 32... done\n");

	return;
}

void test_speed_32(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(32, 32);
	y = mzd_init(32, 32);
	a = mzd_init(32, 32);

	mzd_randomize_custom(x, &my_little_rand, NULL);
	mzd_randomize_custom(a, &my_little_rand, NULL);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va3232_avx(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 32 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va3232_sse(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 32 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va3232_std(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 32 w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va6464_std(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 32(64) w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mzd_mul_m4rm(y, x, a, 0);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	return;
}

/* = 64 = */

void test_correc_64(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *ystd, *ysse, *yavx;

	x = mzd_init(1, 64);
	a = mzd_init(64, 64);
	ym4r = mzd_init(1, 64);
	ystd = mzd_init(1, 64);
	ysse = mzd_init(1, 64);
	yavx = mzd_init(1, 64);

	int avxg = 1;
	int sseg = 1;
	int stdg = 1;

	for (int i = 0; i < tries; i++)
	{
		mzd_randomize_custom(x, &my_little_rand, NULL);
		mzd_randomize_custom(a, &my_little_rand, NULL);

		mul_va6464_avx(yavx, x, a, 1);
		mul_va6464_sse(ysse, x, a, 1);
		mul_va6464_std(ystd, x, a, 1);
		mzd_mul_m4rm(ym4r, x, a, 0);

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
		sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
		stdg = mzd_cmp(ystd, ym4r) == 0 ? stdg : 0;
	}
	printf("[64 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");
	printf("[64 (#%d)] SSE: %s\n", tries, sseg ? "good" : "bad");
	printf("[64 (#%d)] STD: %s\n", tries, stdg ? "good" : "bad");

	return;
}

void test_correc_full_64(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *ystd, *ysse, *yavx;

	for (int c1 = 1; c1 <= 64; c1++)
	{
		for (int r = 1; r <= 64; r++)
		{
			for (int c2 = 1; c2 <= 64; c2++)
			{
				x = mzd_init(r, c1);
				a = mzd_init(c1, c2);
				ym4r = mzd_init(r, c2);
				ystd = mzd_init(r, c2);
				ysse = mzd_init(r, c2);
				yavx = mzd_init(r, c2);

				int avxg = 1;
				int sseg = 1;
				int stdg = 1;

				for (int i = 0; i < tries; i++)
				{
					mzd_randomize_custom(x, &my_little_rand, NULL);
					mzd_randomize_custom(a, &my_little_rand, NULL);

					mul_va6464_avx(yavx, x, a, 1);
					mul_va6464_sse(ysse, x, a, 1);
					mul_va6464_std(ystd, x, a, 1);
					mzd_mul_m4rm(ym4r, x, a, 0);

					avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
					sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
					stdg = mzd_cmp(ystd, ym4r) == 0 ? stdg : 0;
				}
				if (!stdg)
					printf("[%dx%d X %dx%d (#%d)] STD: BAD\n", r, c1, c1, c2, tries);
				if (!sseg)
					printf("[%dx%d X %dx%d (#%d)] SSE: BAD\n", r, c1, c1, c2, tries);
				if (!avxg)
					printf("[%dx%d X %dx%d (#%d)] AVX: BAD\n", r, c1, c1, c2, tries);
			}
		}
	}
	printf("Full 64... done\n");

	return;
}

void test_speed_64(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(64, 64);
	y = mzd_init(64, 64);
	a = mzd_init(64, 64);

	mzd_randomize_custom(x, &my_little_rand, NULL);
	mzd_randomize_custom(a, &my_little_rand, NULL);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va6464_avx(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 64 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va6464_sse(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 64 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va6464_std(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 64 w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mzd_mul_m4rm(y, x, a, 0);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	return;
}

/* = 128 */

void test_correc_128(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *ysse, *yavx;

	x = mzd_init(1, 128);
	a = mzd_init(128, 128);
	ym4r = mzd_init(1, 128);
	ysse = mzd_init(1, 128);
	yavx = mzd_init(1, 128);

	int avxg = 1;
	int sseg = 1;

	for (int i = 0; i < tries; i++)
	{
		mzd_randomize_custom(x, &my_little_rand, NULL);
		mzd_randomize_custom(a, &my_little_rand, NULL);

		mul_va128128_avx(yavx, x, a, 1);
		mul_va128128_sse(ysse, x, a, 1);
		mzd_mul_m4rm(ym4r, x, a, 0);

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
		sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
	}
	printf("[128 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");
	printf("[128 (#%d)] SSE: %s\n", tries, sseg ? "good" : "bad");

	return;
}

void test_correc_full_128(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *ystd, *ysse, *yavx;

	for (int c1 = 65; c1 <= 128; c1++)
	{
		for (int r = 1; r <= 1; r++)
		{
			for (int c2 = 65; c2 <= 128; c2++)
			{
				x = mzd_init(r, c1);
				a = mzd_init(c1, c2);
				ym4r = mzd_init(r, c2);
				ystd = mzd_init(r, c2);
				ysse = mzd_init(r, c2);
				yavx = mzd_init(r, c2);

				int avxg = 1;
				int sseg = 1;
				int stdg = 1;

				for (int i = 0; i < tries; i++)
				{
					mzd_randomize_custom(x, &my_little_rand, NULL);
					mzd_randomize_custom(a, &my_little_rand, NULL);

					mul_va128128_avx(yavx, x, a, 1);
					mul_va128128_sse(ysse, x, a, 1);
					mul_va128128_std(ystd, x, a, 1);
					mzd_mul_m4rm(ym4r, x, a, 0);

					avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
					sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
					stdg = mzd_cmp(ystd, ym4r) == 0 ? stdg : 0;
				}
				if (!stdg)
					printf("[%dx%d X %dx%d (#%d)] STD: BAD\n", r, c1, c1, c2, tries);
				if (!sseg)
					printf("[%dx%d X %dx%d (#%d)] SSE: BAD\n", r, c1, c1, c2, tries);
				if (!avxg)
					printf("[%dx%d X %dx%d (#%d)] AVX: BAD\n", r, c1, c1, c2, tries);
			}
		}
	}
	printf("Full 128... done\n");

	return;
}

void test_speed_128(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(128, 128);
	y = mzd_init(128, 128);
	a = mzd_init(128, 128);

	mzd_randomize_custom(x, &my_little_rand, NULL);
	mzd_randomize_custom(a, &my_little_rand, NULL);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va128128_avx(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 128 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va128128_sse(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 128 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va128128_std(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 128 w/o SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mzd_mul_m4rm(y, x, a, 0);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	return;
}

/* = 256 */

void test_correc_256(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *yavx;

	x = mzd_init(1, 256);
	a = mzd_init(256, 256);
	ym4r = mzd_init(1, 256);
	yavx = mzd_init(1, 256);

	int avxg = 1;

	for (int i = 0; i < tries; i++)
	{
		mzd_randomize_custom(x, &my_little_rand, NULL);
		mzd_randomize_custom(a, &my_little_rand, NULL);

		mul_va256256_avx(yavx, x, a, 1);
		mzd_mul_m4rm(ym4r, x, a, 0);
//		printf("%016llX%016llX%016llX%016llX\n", mzd_read_bits(a, 0, 192, 64), mzd_read_bits(a, 0, 128, 64), mzd_read_bits(a, 0, 64, 64), mzd_read_bits(a, 0, 0, 64));
//		printf("%016llX%016llX%016llX%016llX\n", mzd_read_bits(ym4r, 0, 192, 64), mzd_read_bits(ym4r, 0, 128, 64), mzd_read_bits(ym4r, 0, 64, 64), mzd_read_bits(ym4r, 0, 0, 64));

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
	}
	printf("[256 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");

	return;
}

void test_correc_full_256(int tries)
{
	mzd_t *x, *a;
	mzd_t *ym4r, *yavx, *ysse;

	for (int c1 = 225; c1 <= 256; c1++)
	{
		for (int r = 1; r <= 1; r++)
		{
			for (int c2 = 225; c2 <= 256; c2++)
			{
				x = mzd_init(r, c1);
				a = mzd_init(c1, c2);
				ym4r = mzd_init(r, c2);
				yavx = mzd_init(r, c2);
				ysse = mzd_init(r, c2);

				int avxg = 1;
				int sseg = 1;

				for (int i = 0; i < tries; i++)
				{
					mzd_randomize_custom(x, &my_little_rand, NULL);
					mzd_randomize_custom(a, &my_little_rand, NULL);

					mul_va256256_avx(yavx, x, a, 1);
					mul_224_224_bro_sse2(ysse, x, a, 1);
					mzd_mul_m4rm(ym4r, x, a, 0);

					avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
					sseg = mzd_cmp(ysse, ym4r) == 0 ? sseg : 0;
				}
				if (!avxg)
					printf("[%dx%d X %dx%d (#%d)] AVX: BAD\n", r, c1, c1, c2, tries);
				if (!sseg)
					printf("[%dx%d X %dx%d (#%d)] SSE: BAD\n", r, c1, c1, c2, tries);
			}
		}
	}
	printf("Full 256... done\n");

	return;
}

void test_speed_256(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(256, 256);
	y = mzd_init(256, 256);
	a = mzd_init(256, 256);

	mzd_randomize_custom(x, &my_little_rand, NULL);
	mzd_randomize_custom(a, &my_little_rand, NULL);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va256256_avx(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 256 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_224_224_bro_sse2(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 256 w SSE:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
//		_mzd_mul_va(y, x, a, 1);
		mzd_mul_m4rm(y, x, a, 0);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	return;
}

int main()
{
//	test_correc_matvec_32(1<<20);
//	test_correc_full_32(1<<16);
//	test_correc_32_var(24,32,10);
//	test_speed_32(1 << 20);
//	test_speed_32_var(8, 8, 1 << 24);
//	test_correc_64(1<<20);
//	test_correc_full_64(1<<6);
//	test_speed_64(1 << 20);
//	test_correc_128(1<<20);
	test_correc_full_128(1<<6);
	test_speed_128(1 << 18);
//	test_correc_256(1<<18);
//	test_correc_full_256(1<<9);
//	test_speed_256(1 << 18);

	return 0;
}

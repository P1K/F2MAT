/*
 * ``Fast'' matrix multiplication in GF(2) for small dimensions
 * Uses broadcast-based vectorized algorithms
 * (See e.g. (Käsper and Schwabe, 2009) and (Augot et al., 2014))
 *
 * Pierre Karpman
 * 2017-06
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <immintrin.h>
#include <m4ri/m4ri.h>


// printf("%016llX%016llX\n", _mm_extract_epi64(rr, 0), _mm_extract_epi64(rr, 1));

/* = 32 */

/*
 * All input allocated
 * if clear, reset res before adding the result
 */
void mul_va3232_std(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
//	unsigned arows = A->nrows;
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
//	assert(V->nrows == 1);
//	assert(vcols <= 32);
//	assert(arows == vcols);

	uint32_t m, acc = 0;
	uint32_t v = mzd_read_bits(V, 0, 0, vcols);

	v = ~v;
	for (int i = 0; i < vcols; i++)
	{
		m = (v & 1) - 1;
		acc ^= (m & mzd_read_bits(A, i, 0, acols));
		v >>= 1;
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, acols, 0);
	}
	mzd_xor_bits(res, 0, 0, acols, acc);

	return;
}

/*
 * Same as above, but using SSE4.1
 * (lower version could be used in a rather similar way)
 */
void mul_va3232_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	uint64_t v = mzd_read_bits(V, 0, 0, 32);

	__m128i m2, rr;
	__m128i acc = _mm_setzero_si128();
	__m128i m1  = _mm_set_epi32(1, 2, 4, 8);
	__m128i vv  = _mm_set1_epi32(v);

	for (int i = 0; i < 32; i+=4)
	{
		rr  = _mm_set_epi32(mzd_read_bits(A, i, 0, 32), mzd_read_bits(A, i+1, 0, 32), mzd_read_bits(A, i+2, 0, 32), mzd_read_bits(A, i+3, 0, 32));
		m2  = _mm_and_si128(vv, m1);
		m2  = _mm_cmpeq_epi32(m1, m2);
		rr  = _mm_and_si128(rr, m2);
		acc = _mm_xor_si128(acc, rr);
		m1  = _mm_slli_epi32(m1, 4);
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 32, 0);
	}
	uint32_t tmpacc1 = _mm_extract_epi32(acc, 0) ^ _mm_extract_epi32(acc, 1);
	uint32_t tmpacc2 = _mm_extract_epi32(acc, 2) ^ _mm_extract_epi32(acc, 3);
	tmpacc1 ^= tmpacc2;
	mzd_xor_bits(res, 0, 0, 32, tmpacc1);

	return;
}

/*
 * Same as above, but using AVX2 
 */
void mul_va3232_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	uint64_t v = mzd_read_bits(V, 0, 0, 32);

	__m256i m2, rr;
	__m256i acc = _mm256_setzero_si256();
	__m256i m1  = _mm256_set_epi32(1, 2, 4, 8, 16, 32, 64, 128);
	__m256i vv  = _mm256_set1_epi32(v);

	for (int i = 0; i < 32; i+=8)
	{
		rr  = _mm256_set_epi32(mzd_read_bits(A, i, 0, 64), mzd_read_bits(A, i+1, 0, 64), mzd_read_bits(A, i+2, 0, 64), mzd_read_bits(A, i+3, 0, 64),
				mzd_read_bits(A, i+4, 0, 64), mzd_read_bits(A, i+5, 0, 64), mzd_read_bits(A, i+6, 0, 64), mzd_read_bits(A, i+7, 0, 64));
		m2  = _mm256_and_si256(vv, m1);
		m2  = _mm256_cmpeq_epi32(m1, m2);
		rr  = _mm256_and_si256(rr, m2);
		acc = _mm256_xor_si256(acc, rr);
		m1  = _mm256_slli_epi32(m1, 8);
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 32, 0);
	}
	uint32_t tmpacc1 = _mm256_extract_epi32(acc, 0) ^ _mm256_extract_epi32(acc, 1);
	uint32_t tmpacc2 = _mm256_extract_epi32(acc, 2) ^ _mm256_extract_epi32(acc, 3);
	uint32_t tmpacc3 = _mm256_extract_epi32(acc, 4) ^ _mm256_extract_epi32(acc, 5);
	uint32_t tmpacc4 = _mm256_extract_epi32(acc, 6) ^ _mm256_extract_epi32(acc, 7);
	tmpacc1 ^= tmpacc2;
	tmpacc3 ^= tmpacc4;
	tmpacc1 ^= tmpacc3;
	mzd_xor_bits(res, 0, 0, 32, tmpacc1);

	return;
}

/* = 64 */

/*
 * All input allocated
 * V a row vector of dim 64
 * A a square matrix of dim 64
 * if clear, reset res before adding the result
 */
void mul_va6464_std(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	uint64_t m, acc = 0;
	uint64_t v = mzd_read_bits(V, 0, 0, 64);

	v = ~v;
	for (int i = 0; i < 64; i++)
	{
		m = (v & 1) - 1ull;
		acc ^= (m & mzd_read_bits(A, i, 0, 64));
//		acc ^= (m & *(A->rows[i])); // dirty, not faster 
		v >>= 1;
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 64, 0);
	}
	mzd_xor_bits(res, 0, 0, 64, acc);

	return;
}

/*
 * Same as above, but using SSE4.1
 * (lower version could be used in a rather similar way)
 */
void mul_va6464_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	uint64_t v = mzd_read_bits(V, 0, 0, 64);

	__m128i m2, rr;
	__m128i acc = _mm_setzero_si128();
	__m128i m1  = _mm_set_epi64x(1, 2);
	__m128i vv  = _mm_set1_epi64x(v);

	for (int i = 0; i < 64; i+=2)
	{
		rr  = _mm_set_epi64x(mzd_read_bits(A, i, 0, 64), mzd_read_bits(A, i+1, 0, 64));
//		rr = _mm_loadu_si128((__m128i*)A->rows[i]); // unfortunately, the matrix structure doesn't allow to do this much faster load
		m2  = _mm_and_si128(vv, m1);
		m2  = _mm_cmpeq_epi64(m1, m2);
		rr  = _mm_and_si128(rr, m2);
		acc = _mm_xor_si128(acc, rr);
		m1  = _mm_slli_epi64(m1, 2);
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 64, 0);
	}
	mzd_xor_bits(res, 0, 0, 64, _mm_extract_epi64(acc, 0));
	mzd_xor_bits(res, 0, 0, 64, _mm_extract_epi64(acc, 1));

	return;
}

/*
 * Same as above, but using AVX2 
 */
void mul_va6464_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	uint64_t v = mzd_read_bits(V, 0, 0, 64);

	__m256i m2, rr;
	__m256i acc = _mm256_setzero_si256();
	__m256i m1  = _mm256_set_epi64x(1, 2, 4, 8);
	__m256i vv  = _mm256_set1_epi64x(v);

	for (int i = 0; i < 64; i+=4)
	{
		rr  = _mm256_set_epi64x(mzd_read_bits(A, i, 0, 64), mzd_read_bits(A, i+1, 0, 64), mzd_read_bits(A, i+2, 0, 64), mzd_read_bits(A, i+3, 0, 64));
//		rr  = _mm256_set_epi64x(*(A->rows[i]), *(A->rows[i+1]), *(A->rows[i+2]), *(A->rows[i+3])); // dirty, not faster
//		rr  = _mm256_loadu_si256((__m256i*)(A->rows[i])); // unfortunately, the matrix structure doesn't allow to do this much faster load
		m2  = _mm256_and_si256(vv, m1);
		m2  = _mm256_cmpeq_epi64(m1, m2);
		rr  = _mm256_and_si256(rr, m2);
		acc = _mm256_xor_si256(acc, rr);
		m1  = _mm256_slli_epi64(m1, 4);
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 64, 0);
	}
	uint64_t tmpacc1 = _mm256_extract_epi64(acc, 0) ^ _mm256_extract_epi64(acc, 1);
	uint64_t tmpacc2 = _mm256_extract_epi64(acc, 2) ^ _mm256_extract_epi64(acc, 3);
	tmpacc1 ^= tmpacc2;
	mzd_xor_bits(res, 0, 0, 64, tmpacc1);

	return;
}

/* = 128 = */

/* Could do one w/o SSE too, but well... */

void mul_va128128_sse(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	__m128i v  = _mm_set_epi64x(mzd_read_bits(V, 0, 64, 64), mzd_read_bits(V, 0, 0, 64));

	__m128i m2, m3, rr2, rr3;
	__m128i acc = _mm_setzero_si128();
	__m128i m1  = _mm_set1_epi64x(1);

	for (int i = 0; i < 64; i++)
	{
		rr2  = _mm_set_epi64x(mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
		rr3  = _mm_set_epi64x(mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
//		rr2 = _mm_loadu_si128((__m128i*)A->rows[i]); // now fine but not really faster
//		rr3 = _mm_loadu_si128((__m128i*)A->rows[i+64]); // now fine but not really faster
		m2  = _mm_and_si128(v, m1);
		// there's a tradeoff between comp granularity and code/masks size
		m2  = _mm_cmpeq_epi64(m1, m2);
		m3  = _mm_shuffle_epi32(m2,0xFF);
		m2  = _mm_shuffle_epi32(m2,0x00);
		rr2 = _mm_and_si128(rr2, m2);
		rr3 = _mm_and_si128(rr3, m3);
		acc = _mm_xor_si128(acc, rr2);
		acc = _mm_xor_si128(acc, rr3);
		m1  = _mm_slli_epi64(m1, 1);
	}

	if (clear)
	{
		mzd_and_bits(res, 0, 0, 64, 0);
		mzd_and_bits(res, 0, 64, 64, 0);
	}
	mzd_xor_bits(res, 0, 0, 64, _mm_extract_epi64(acc, 0));
	mzd_xor_bits(res, 0, 64, 64, _mm_extract_epi64(acc, 1));

	return;
}

void mul_va128128_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	__m128i v  = _mm_set_epi64x(mzd_read_bits(V, 0, 64, 64), mzd_read_bits(V, 0, 0, 64));

	__m256i m2, m3, rr2, rr3;
	__m256i acc = _mm256_setzero_si256();
	__m256i m1  = _mm256_set_epi64x(2, 2, 1, 1);
	__m256i vv  = _mm256_set_m128i(v, v);

	for (int i = 0; i < 64; i+=2)
	{
//		rr2 = _mm256_set_epi64x(mzd_read_bits(A, i+1, 64, 64), mzd_read_bits(A, i+1, 0, 64), mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
//		rr3 = _mm256_set_epi64x(mzd_read_bits(A, i+65, 64, 64), mzd_read_bits(A, i+65, 0, 64), mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
		rr2 = _mm256_loadu_si256((__m256i*)A->rows[i]); // makes an assumption on the matrix representation, but quite faster
		rr3 = _mm256_loadu_si256((__m256i*)A->rows[i+64]); // ditto
		m2  = _mm256_and_si256(vv, m1);
		m2  = _mm256_cmpeq_epi64(m1, m2);
		m3  = _mm256_shuffle_epi32(m2,0xFF);
		m2  = _mm256_shuffle_epi32(m2,0x00);
		rr2 = _mm256_and_si256(rr2, m2);
		rr3 = _mm256_and_si256(rr3, m3);
		acc = _mm256_xor_si256(acc, rr2);
		acc = _mm256_xor_si256(acc, rr3);
		m1  = _mm256_slli_epi64(m1, 2);
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 64, 0);
		mzd_and_bits(res, 0, 64, 64, 0);
	}
	mzd_xor_bits(res, 0, 0, 64, _mm256_extract_epi64(acc, 0));
	mzd_xor_bits(res, 0, 64, 64, _mm256_extract_epi64(acc, 1));
	mzd_xor_bits(res, 0, 0, 64, _mm256_extract_epi64(acc, 2));
	mzd_xor_bits(res, 0, 64, 64, _mm256_extract_epi64(acc, 3));

	return;
}

/* = 256 = */

void mul_va256256_avx(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	__m256i v  = _mm256_set_epi64x(mzd_read_bits(V, 0, 192, 64), mzd_read_bits(V, 0, 128, 64), mzd_read_bits(V, 0, 64, 64), mzd_read_bits(V, 0, 0, 64));

	__m256i m2, m3, m4, m5, rr2, rr3, rr4, rr5;
	__m256i acc = _mm256_setzero_si256();
	__m256i m1  = _mm256_set1_epi64x(1);

	for (int i = 0; i < 64; i++)
	{
		rr2 = _mm256_set_epi64x(mzd_read_bits(A, i, 192, 64), mzd_read_bits(A, i, 128, 64), mzd_read_bits(A, i, 64, 64), mzd_read_bits(A, i, 0, 64));
		rr3 = _mm256_set_epi64x(mzd_read_bits(A, i+64, 192, 64), mzd_read_bits(A, i+64, 128, 64), mzd_read_bits(A, i+64, 64, 64), mzd_read_bits(A, i+64, 0, 64));
		rr4 = _mm256_set_epi64x(mzd_read_bits(A, i+128, 192, 64), mzd_read_bits(A, i+128, 128, 64), mzd_read_bits(A, i+128, 64, 64), mzd_read_bits(A, i+128, 0, 64));
		rr5 = _mm256_set_epi64x(mzd_read_bits(A, i+192, 192, 64), mzd_read_bits(A, i+192, 128, 64), mzd_read_bits(A, i+192, 64, 64), mzd_read_bits(A, i+192, 0, 64));
//		rr2 = _mm256_loadu_si256((__m256i*)A->rows[i]); // now fine but not really faster
//		rr3 = _mm256_loadu_si256((__m256i*)A->rows[i+64]);
//		rr4 = _mm256_loadu_si256((__m256i*)A->rows[i+128]);
//		rr5 = _mm256_loadu_si256((__m256i*)A->rows[i+192]);
		m2  = _mm256_and_si256(v, m1);
		// ditto, tradeoff
		m2  = _mm256_cmpeq_epi64(m1, m2);
		m5  = _mm256_permute4x64_epi64(m2, 0xFF); 
		m4  = _mm256_permute4x64_epi64(m2, 0xAA); 
		m3  = _mm256_permute4x64_epi64(m2, 0x55); 
		m2  = _mm256_permute4x64_epi64(m2, 0x00); 
		rr2 = _mm256_and_si256(rr2, m2);
		rr3 = _mm256_and_si256(rr3, m3);
		rr4 = _mm256_and_si256(rr4, m4);
		rr5 = _mm256_and_si256(rr5, m5);
		rr2 = _mm256_xor_si256(rr3, rr2);
		rr4 = _mm256_xor_si256(rr5, rr4);
		acc = _mm256_xor_si256(acc, rr2);
		acc = _mm256_xor_si256(acc, rr4);
		m1  = _mm256_slli_epi64(m1, 1);
	}
	if (clear)
	{
		mzd_and_bits(res, 0, 0, 64, 0);
		mzd_and_bits(res, 0, 64, 64, 0);
		mzd_and_bits(res, 0, 128, 64, 0);
		mzd_and_bits(res, 0, 192, 64, 0);
	}
	mzd_xor_bits(res, 0, 0, 64, _mm256_extract_epi64(acc, 0));
	mzd_xor_bits(res, 0, 64, 64, _mm256_extract_epi64(acc, 1));
	mzd_xor_bits(res, 0, 128, 64, _mm256_extract_epi64(acc, 2));
	mzd_xor_bits(res, 0, 192, 64, _mm256_extract_epi64(acc, 3));

	return;
}

/*
 * TESTS
 */

/* = 32 = */

void test_correc_32(int tries)
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
		mzd_randomize(x);
		mzd_randomize(a);

		mul_va3232_avx(yavx, x, a, 1);
		mul_va3232_sse(ysse, x, a, 1);
		mul_va3232_std(ystd, x, a, 1);
		_mzd_mul_va(ym4r, x, a, 1);

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
		sseg = mzd_cmp(ysse, ym4r) == 0 ? avxg : 0;
		stdg = mzd_cmp(ystd, ym4r) == 0 ? avxg : 0;
//		printf("%08llX\n", mzd_read_bits(y, 0, 0, 32));
	}
	printf("[32 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");
	printf("[32 (#%d)] SSE: %s\n", tries, sseg ? "good" : "bad");
	printf("[32 (#%d)] STD: %s\n", tries, stdg ? "good" : "bad");

	return;
}

//void test_correc_32_var(int dim1, int dim2, int tries)
//{
//	mzd_t *x, *y, *a;
//
//	x = mzd_init(1, dim2);
//	y = mzd_init(1, dim2);
//	a = mzd_init(dim2, dim1);
//
//	for (int i = 0; i < tries; i++)
//	{
//		mzd_randomize(x);
//		mzd_randomize(a);
//
//		mul_va3232_std(y, x, a, 1);
//		printf("%08llX\n", mzd_read_bits(y, 0, 0, dim1));
//		_mzd_mul_va(y, x, a, 1);
//		printf("%08llX\n", mzd_read_bits(y, 0, 0, dim1));
//
//		printf("\n");
//	}
//
//	return;
//}

void test_speed_32(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(1, 32);
	y = mzd_init(1, 32);
	a = mzd_init(32, 32);

	mzd_randomize(x);
	mzd_randomize(a);
	
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
		_mzd_mul_va(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	return;
}

void test_speed_32_var(int dim1, int dim2, unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(1, dim2);
	y = mzd_init(1, dim2);
	a = mzd_init(dim2, dim1);

	mzd_randomize(x);
	mzd_randomize(a);
	
	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va3232_std(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' %dx%d w/o SSE:\t %llu usecs (#%u)\n", dim2, dim1, tusec, iter);
	
	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		_mzd_mul_va(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("M4RI:\t\t\t %llu usecs (#%u)\n", tusec, iter);

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
		mzd_randomize(x);
		mzd_randomize(a);

		mul_va6464_avx(yavx, x, a, 1);
		mul_va6464_sse(ysse, x, a, 1);
		mul_va6464_std(ystd, x, a, 1);
		_mzd_mul_va(ym4r, x, a, 1);

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
		sseg = mzd_cmp(ysse, ym4r) == 0 ? avxg : 0;
		stdg = mzd_cmp(ystd, ym4r) == 0 ? avxg : 0;
	}
	printf("[64 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");
	printf("[64 (#%d)] SSE: %s\n", tries, sseg ? "good" : "bad");
	printf("[64 (#%d)] STD: %s\n", tries, stdg ? "good" : "bad");

	return;
}

void test_speed_64(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(1, 64);
	y = mzd_init(1, 64);
	a = mzd_init(64, 64);

	mzd_randomize(x);
	mzd_randomize(a);
	
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
		_mzd_mul_va(y, x, a, 1);
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
		mzd_randomize(x);
		mzd_randomize(a);

		mul_va128128_avx(yavx, x, a, 1);
		mul_va128128_sse(ysse, x, a, 1);
		_mzd_mul_va(ym4r, x, a, 1);

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
		sseg = mzd_cmp(ysse, ym4r) == 0 ? avxg : 0;
	}
	printf("[128 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");
	printf("[128 (#%d)] SSE: %s\n", tries, sseg ? "good" : "bad");

	return;
}

void test_speed_128(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(1, 128);
	y = mzd_init(1, 128);
	a = mzd_init(128, 128);

	mzd_randomize(x);
	mzd_randomize(a);
	
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
		_mzd_mul_va(y, x, a, 1);
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
		mzd_randomize(x);
		mzd_randomize(a);

		mul_va256256_avx(yavx, x, a, 1);
		_mzd_mul_va(ym4r, x, a, 1);
//		printf("%016llX%016llX%016llX%016llX\n", mzd_read_bits(y, 0, 192, 64), mzd_read_bits(y, 0, 128, 64), mzd_read_bits(y, 0, 64, 64), mzd_read_bits(y, 0, 0, 64));

		avxg = mzd_cmp(yavx, ym4r) == 0 ? avxg : 0;
	}
	printf("[256 (#%d)] AVX: %s\n", tries, avxg ? "good" : "bad");

	return;
}

void test_speed_256(unsigned iter)
{
	struct timeval tv1, tv2;
	uint64_t tusec;
	mzd_t *x, *y, *a;

	x = mzd_init(1, 256);
	y = mzd_init(1, 256);
	a = mzd_init(256, 256);

	mzd_randomize(x);
	mzd_randomize(a);
	
	gettimeofday(&tv1, NULL);
	for (unsigned i = 0; i < iter; i++)
		mul_va256256_avx(y, x, a, 1);
	gettimeofday(&tv2, NULL);
	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
	printf("`Fast' 256 w AVX:\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);
	
//	gettimeofday(&tv1, NULL);
//	for (unsigned i = 0; i < iter; i++)
//		_mzd_mul_va(y, x, a, 1);
//	gettimeofday(&tv2, NULL);
//	tusec = ((1000000*tv2.tv_sec + tv2.tv_usec) - (1000000*tv1.tv_sec + tv1.tv_usec));
//	printf("M4RI:\t\t\t %llu usecs (#%u) [%f usecs/op]\n", tusec, iter, (double)tusec / (double)iter);

	return;
}

int main()
{
	test_correc_32(1<<20);
//	test_correc_32_var(24,32,10);
//	test_speed_32(1 << 28);
//	test_speed_32_var(8, 8, 1 << 24);
	test_correc_64(1<<20);
//	test_speed_64(1 << 28);
	test_correc_128(1<<20);
//	test_speed_128(1 << 28);
	test_correc_256(1<<20);
//	test_speed_256(1 << 24);

	return 0;
}

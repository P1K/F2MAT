#include "bro_ska.h"


/*
 * assert(V->ncols <= 64)
 * assert(A->ncols <= 64)
 */
void mul_64_64_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;

	for (int j = 0; j < vrows; j++)
	{
		uint64_t m, acc = 0;
		uint64_t v = mzd_read_bits(V, j, 0, vcols);

		v = ~v;
		for (int i = 0; i < vcols; i++)
		{
			m = (v & 1) - 1;
			// up to (64 - acols) bits of junk may be accumulated
			// but they're eventually not copied
			acc ^= (m & mzd_read_bits(A, i, 0, 64));
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
 * assert(V->ncols > 64)
 * assert(V->ncols <= 128)
 * assert(A->ncols <= 64)
 */
void mul_128_64_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;

	for (int j = 0; j < vrows; j++)
	{
		uint64_t mlo, mhi;
		uint64_t acc0 = 0;
		uint64_t acc1 = 0;
		uint64_t vlo = mzd_read_bits(V, j, 0, 64);
		uint64_t vhi = mzd_read_bits(V, j, 64, vcols-64);

		vlo = ~vlo;
		vhi = ~vhi;
		int i = 0;
		for (; i < 64; i++)
		{
			mlo = (vlo & 1) - 1;
			// up to (64 - acols) bits of junk may be accumulated
			// but they're eventually not copied
			acc0 ^= (mlo & mzd_read_bits(A, i, 0, 64));
			vlo >>= 1;
		}
		for (; i < vcols; i++)
		{
			mhi = (vhi & 1) - 1;
			acc1 ^= (mhi & mzd_read_bits(A, i, 0, 64)); // ditto
			vhi >>= 1;
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, acols, 0);
		}
		mzd_xor_bits(res, j, 0, acols, acc0);
		mzd_xor_bits(res, j, 0, acols, acc1);
	}

	return;
}

/*
 * assert(V->ncols <= 64)
 * assert(A->ncols > 64)
 * assert(A->ncols <= 128)
 */
void mul_64_128_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;

	for (int j = 0; j < vrows; j++)
	{
		uint64_t m;
		uint64_t acclo = 0;
		uint64_t acchi = 0;
		uint64_t v = mzd_read_bits(V, j, 0, vcols);

		v = ~v;
		for (int i = 0; i < vcols; i++)
		{
			m = (v & 1) - 1;
			acclo ^= (m & mzd_read_bits(A, i, 0, 64));
			// up to (128 - acols) bits of junk may be accumulated
			// but they're eventually not copied
			acchi ^= (m & mzd_read_bits(A, i, 64, 64));
			v >>= 1;
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
			mzd_and_bits(res, j, 64, acols-64, 0);
		}
		mzd_xor_bits(res, j, 0, 64, acclo);
		mzd_xor_bits(res, j, 64, acols-64, acchi);
	}

	return;
}

/*
 * assert(V->ncols > 64)
 * assert(V->ncols <= 128)
 * assert(A->ncols > 64)
 * assert(A->ncols <= 128)
 */
void mul_128_128_bro_ska(mzd_t *res, mzd_t *V, mzd_t *A, int clear)
{
	unsigned acols = A->ncols;
	unsigned vcols = V->ncols;
	unsigned vrows = V->nrows;

	for (int j = 0; j < vrows; j++)
	{
		uint64_t mlo, mhi;
		uint64_t acc0lo = 0;
		uint64_t acc0hi = 0;
		uint64_t acc1lo = 0;
		uint64_t acc1hi = 0;
		uint64_t vlo = mzd_read_bits(V, j, 0, 64);
		uint64_t vhi = mzd_read_bits(V, j, 64, vcols-64);

		vlo = ~vlo;
		vhi = ~vhi;
		int i = 0;
		for (; i < 64; i++)
		{
			mlo = (vlo & 1) - 1;
			acc0lo ^= (mlo & mzd_read_bits(A, i, 0, 64));
			// up to (128 - acols) bits of junk may be accumulated
			// but they're eventually not copied
			acc0hi ^= (mlo & mzd_read_bits(A, i, 64, 64));
			vlo >>= 1;
		}
		for (; i < vcols; i++)
		{
			mhi = (vhi & 1) - 1;
			acc1lo ^= (mhi & mzd_read_bits(A, i, 0, 64));
			acc1hi ^= (mhi & mzd_read_bits(A, i, 64, 64)); // ditto
			vhi >>= 1;
		}
		if (clear)
		{
			mzd_and_bits(res, j, 0, 64, 0);
			mzd_and_bits(res, j, 64, acols-64, 0);
		}
		mzd_xor_bits(res, j, 0, 64, acc0lo);
		mzd_xor_bits(res, j, 0, 64, acc1lo);
		mzd_xor_bits(res, j, 64, acols-64, acc0hi);
		mzd_xor_bits(res, j, 64, acols-64, acc1hi);
	}

	return;
}

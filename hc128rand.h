/*
 * Portable PRNG with static state for easy randomness, using HC-128
 * Adapted from Hongjun Wu's reference (one-step) implementation
 * (http://www3.ntu.edu.sg/home/wuhj/research/hc/index.html)
 * (See also the original notice below)
 *
 * NOT THREAD SAFE!
 *
 * PK, 2016
 */

#ifndef __HC_128_RANDOM
#define __HC_128_RANDOM

#include <stdio.h>
#include <stdint.h>

/* This program gives the reference implementation of stream cipher HC-128
 *
 * HC-128 is a final portfolio cipher of eSTREAM, of the European Network of
 * Excellence for Cryptology (ECRYPT, 2004-2008).
 * The docuement of HC-128 is available at:
 * 1) Hongjun Wu. ``The Stream Cipher HC-128.'' New Stream Cipher Designs -- The eSTREAM Finalists, LNCS 4986, pp. 39-47, Springer-Verlag, 2008.
 * 2) eSTREAM website:  http://www.ecrypt.eu.org/stream/hcp3.html
 *
 *  ------------------------------------
 *  Performance of this non-optimized implementation:
 *
 * Microprocessor: Intel CORE 2 processor (Core 2 Duo Mobile P9400 2.53GHz)
 * Operating System: 32-bit Debian 5.0 (Linux kernel 2.6.26-2-686)
 * Speed of encrypting long message:
 * 1) 6.3 cycle/byte   compiler: Intel C++ compiler 11.1   compilation option: icc -O2
 * 2) 3.8 cycles/byte  compiler: gcc 4.3.2                 compilation option: gcc -O3
 *
 * Microprocessor: Intel CORE 2 processor (Core 2 Quad Q6600 2.4GHz)
 * Operating System: 32-bit Windows Vista Business
 * Speed of encrypting long message:
 * 1) 6.2 cycles/byte  compiler: Intel C++ compiler 11.1    compilation option: icl /O2
 * 2) 6.4 cycles/byte  compiler: Microsoft Visual C++ 2008  compilation option: release
 *
 * ------------------------------------
 * Written by: Hongjun Wu
 * Last Modified: December 15, 2009
 */

#include <stdio.h>
#include <stdint.h>

typedef struct {
	uint32_t P[512];
	uint32_t Q[512];
	uint32_t counter1024;     /*counter1024 = i mod 1024 */
	uint32_t keystreamword;   /*a 32-bit keystream word*/
} HC128_State;

static int __my_little_init_was_done = 0;
static HC128_State __my_little_hc128_state;

#define ROTR32(x,n)   ( ((x) >> (n))  | ((x) << (32 - (n))) )
#define ROTL32(x,n)   ( ((x) << (n))  | ((x) >> (32 - (n))) )

#define f1(x)    (ROTR32((x),7) ^ ROTR32((x),18) ^ ((x) >> 3))
#define f2(x)    (ROTR32((x),17) ^ ROTR32((x),19) ^ ((x) >> 10))

/*g1 and g2 functions as defined in the HC-128 document*/
#define g1(x,y,z)  ((ROTR32((x),10)^ROTR32((z),23))+ROTR32((y),8))
#define g2(x,y,z)  ((ROTL32((x),10)^ROTL32((z),23))+ROTL32((y),8))

/*function h1*/
uint32_t h1(HC128_State *state, uint32_t u) {
	uint32_t tem; 
	uint8_t  a,c;
	a = (uint8_t) ((u));
	c = (uint8_t) ((u) >> 16);
	tem = state->Q[a]+state->Q[256+c];
	return (tem);
}

/*function h2*/
uint32_t h2(HC128_State *state, uint32_t u) {
	uint32_t tem; 
	uint8_t  a,c;
	a = (uint8_t) ((u));
	c = (uint8_t) ((u) >> 16);
	tem = state->P[a]+state->P[256+c];
	return (tem);
}

/* one step of HC-128:
 * state is updated;
 * a 32-bit keystream word is generated and stored in "state->keystreamword";
 */
void __my_little_hc128_step(HC128_State *state)
{
	uint32_t i,i3, i10, i12, i511;

	i   = state->counter1024 & 0x1ff;
	i3  = (i - 3) & 0x1ff;
	i10 = (i - 10) & 0x1ff;
	i12 = (i - 12) & 0x1ff;
	i511 = (i - 511) & 0x1ff;

	if (state->counter1024 < 512) {
		state->P[i] = state->P[i] + g1(state->P[i3],state->P[i10],state->P[i511]);
		state->keystreamword = h1(state,state->P[i12]) ^ state->P[i];
	}
	else {
		state->Q[i] = state->Q[i] + g2(state->Q[i3],state->Q[i10],state->Q[i511]);
		state->keystreamword = h2(state,state->Q[i12]) ^ state->Q[i];
	}
	state->counter1024 = (state->counter1024+1) & 0x3ff;
}


/* one step of HC-128 in the intitalization stage:
 * a 32-bit keystream word is generated to update the state;
 */
void __my_little_hc128_init_step(HC128_State *state)
{
	uint32_t i,i3, i10, i12, i511;

	i   = state->counter1024 & 0x1ff;
	i3  = (i - 3) & 0x1ff;
	i10 = (i - 10) & 0x1ff;
	i12 = (i - 12) & 0x1ff;
	i511 = (i - 511) & 0x1ff;

	if (state->counter1024 < 512) {
		state->P[i] = state->P[i] + g1(state->P[i3],state->P[i10],state->P[i511]);
		state->P[i] = h1(state,state->P[i12]) ^ state->P[i];
	}
	else {
		state->Q[i] = state->Q[i] + g2(state->Q[i3],state->Q[i10],state->Q[i511]);
		state->Q[i] = h2(state,state->Q[i12]) ^ state->Q[i];
	}
	state->counter1024 = (state->counter1024+1) & 0x3ff;
}


/* this function initialize the state using 128-bit key and 128-bit IV */
void __my_little_hc128_initialization(HC128_State *state, uint8_t key[16], uint8_t iv[16])
{

	uint32_t W[1024+256],i;

	/* expand the key and iv into the state */

	for (i = 0; i < 4; i++) {W[i] = ((uint32_t*)key)[i]; W[i+4] = ((uint32_t*)key)[i];}
	for (i = 0; i < 4; i++) {W[i+8] = ((uint32_t*)iv)[i]; W[i+12] = ((uint32_t*)iv)[i];}

	for (i = 16; i < 1024+256; i++) W[i] = f2(W[i-2]) + W[i-7] + f1(W[i-15]) + W[i-16]+i;

	for (i = 0; i < 512; i++)  state->P[i] = W[i+256];
	for (i = 0; i < 512; i++)  state->Q[i] = W[i+256+512];

	state->counter1024 = 0;

	/* update the cipher for 1024 steps without generating output */
	for (i = 0; i < 1024; i++)  __my_little_hc128_init_step(state);
}

/* This function initializes one state with a zero IV and a key from /dev/urandom */
void __my_little_hc128_unseeded_init(HC128_State *state)
{
	uint8_t key[16];
	uint8_t iv [16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	uint8_t zk [16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	FILE *urd = fopen("/dev/urandom", "r");

	if (urd == NULL)
	{
		fprintf(stderr, "failed to initialize the little hc128 prng [No file called /dev/urandom]\n");
		__my_little_hc128_initialization(state, zk, iv);
		return;
	}

	if(1 != fread(key, 16, 1, urd))
	{
		fprintf(stderr, "failed to initialize the little hc128 prng [Not enough p$ bytes]\n");
	}
	fclose(urd);
	__my_little_hc128_initialization(state, key, iv);
	return;
}

/* This function initializes the static state with a zero IV and a user-provided 128-bit key */
void __my_little_hc128_seeded_init(uint8_t key[16])
{
	uint8_t iv [16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	__my_little_hc128_initialization(&__my_little_hc128_state, key, iv);
	__my_little_init_was_done = 1;
	return;
}

/* This function runs one step of HC128 on a static state and returns a 32-bit random */
uint32_t __my_little_hc128_random(void)
{
	if (!__my_little_init_was_done)
	{
		__my_little_hc128_unseeded_init(&__my_little_hc128_state);
		__my_little_init_was_done = 1;
	}

	__my_little_hc128_step(&__my_little_hc128_state);

	return (__my_little_hc128_state.keystreamword);
}

/* Same as above, without checking that the state was initialized */
uint32_t __my_little_hc128_unsafe_random(void)
{
	__my_little_hc128_step(&__my_little_hc128_state);

	return (__my_little_hc128_state.keystreamword);
}

/*
 * Aliases
 */

uint32_t hc128random(void)
{
	return __my_little_hc128_random();
}

uint32_t hc128random_unsafe(void)
{
	return __my_little_hc128_unsafe_random();
}

void hc128random_set(uint8_t seed[16])
{
	__my_little_hc128_seeded_init(seed);
}
#endif

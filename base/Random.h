#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <cstdlib>


unsigned long long int* next_random;
extern long int workThreads;


extern "C"
void randReset(long int seed)
{
	next_random = (unsigned long long*) calloc(workThreads, sizeof(unsigned long long));
	for (long int i = 0; i < workThreads; i++)
		next_random[i] = seed;
}


unsigned long long int randd(long int id)
{
	auto& r = next_random[id];
	r *= (unsigned long long) 25214903917;
	r += 11;
	return r;
}


long int rand_max(long int id, long int x)
{
	long int res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

#endif

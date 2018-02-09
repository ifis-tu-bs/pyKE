#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <vector> // std::vector


std::vector<uint64_t> next_random;


extern "C"
void randReset(uint64_t workers, uint64_t seed)
{
	next_random.resize(workers);
	for (auto& i: next_random)
		i = seed;
}


uint64_t randd(uint64_t id)
{
	auto& r = next_random.at(id);
	r *= (uint64_t)25214903917;
	r += 11;
	return r;
}


uint64_t rand_max(uint64_t id, uint64_t x)
{
	uint64_t res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}


#endif // RANDOM_H

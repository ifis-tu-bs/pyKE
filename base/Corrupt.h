#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Triple.h"
#include "Reader.h"
#include <algorithm> // std::equal_range


	/*
Answers a question with a randomized unknown head.
	*/
ent_id corrupt_head(
		unsigned long int id,
		ent_id tail,
		rel_id relation)
{
	// using precalculated better range
	const auto range = std::equal_range(lefHead[tail], rigHead[tail], Triple(0, tail, relation), Triple::cmp_r);
	const auto lower = range.first;
	const auto upper = range.second;

	const ent_id x = rand_max(id, entityTotal - (upper - lower));
	if (x < lower->t)
		return x;
	if (x + (upper - lower) > (upper-1)->t)
		return x + (upper - lower);

	// l <- min y s.t. x + y > A[y+1].h
	auto l = lower;
	auto u = upper;
	while (u - l > 1)
	{
		const auto m = l + ((u - l) >> 1);
		if ((m+1)->t < x + (m - l))
			l = m;
		else
			u = m;
	}
	return x + (l - lower);
}


	/*
Answers a question with a randomized unknown tail.
	*/
ent_id corrupt_tail(unsigned long int id, ent_id head, rel_id relation)
{
	// using precalculated better range
	const auto range = std::equal_range(lefTail[head], rigTail[head], Triple(head, 0, relation), Triple::cmp_r);
	const auto lower = range.first;
	const auto upper = range.second;

	const ent_id x = rand_max(id, entityTotal - (upper - lower));
	if (x < lower->h)
		return x;
	if (x + (upper - lower) > (upper-1)->h)
		return x + (upper - lower);

	// l <- min y s.t. x + y > A[y+1].h
	auto l = lower;
	auto u = upper;
	while (u - l > 1)
	{
		const auto m = l + ((u - l) >> 1);
		if ((m+1)->h < x + (m - l))
			l = m;
		else
			u = m;
	}
	return x + (l - lower);
}


	/*
Answers a question with a randomized unknown relation.
	*/
rel_id corrupt_rel(unsigned long int id, ent_id head, ent_id tail)
{
	// using precalculated better range
	const auto range = std::equal_range(lefRel[head], rigHead[head], Triple(head, tail, 0), Triple::cmp_t);
	const auto lower = range.first;
	const auto upper = range.second;

	const ent_id x = rand_max(id, relationTotal - (upper - lower));
	if (x < lower->h)
		return x;
	if (x + (upper - lower) > (upper-1)->h)
		return x + (upper - lower);

	// l <- min y s.t. x + y > A[y+1].h
	auto l = lower;
	auto u = upper;
	while (u - l > 1)
	{
		const auto m = l + ((u - l) >> 1);
		if ((m+1)->h < x + (m - l))
			l = m;
		else
			u = m;
	}
	return x + (l - lower);
}

#endif

#ifndef TEST_H
#	define TEST_H
#	include "Setting.h"
#	include "Reader.h"
#	include "Triple.h"
#	include <algorithm> // std::binary_search


extern "C" void query_head(char*, int64_t, int64_t);
extern "C" void query_tail(int64_t, char*, int64_t);
extern "C" void query_rel(int64_t, int64_t, char*);


extern "C" void query_head(
		char* out,
		int64_t tail,
		int64_t relation)
try
{
	// assuming out has size `entityTotal` and is zero-initialized
	const auto range = std::equal_range(lefHead.at(tail), rigHead.at(tail),
			Triple(0, tail, relation), Triple::cmp_r);
	const auto begin = range.first;
	const auto end = range.second;
	for (auto i = begin; i != end; ++i)
		out[i->h] = 1;
}
catch (std::out_of_range& e)
{
}


extern "C" void query_tail(
		int64_t head,
		char* out,
		int64_t relation)
try
{
	// assuming out has size `entityTotal` and is zero-initialized
	const auto range = std::equal_range(lefTail.at(head), rigTail.at(head),
			Triple(head, 0, relation), Triple::cmp_r);
	const auto begin = range.first;
	const auto end = range.second;
	for (auto i = begin; i != end; ++i)
		out[i->t] = 1;
}
catch (std::out_of_range& e)
{
}


extern "C" void query_rel(
		int64_t head,
		int64_t tail,
		char* out)
try
{
	// assuming out has size `relationTotal` and is zero-initialized
	const auto range = std::equal_range(lefRel.at(head), rigRel.at(head),
			Triple(head, tail, 0), Triple::cmp_t);
	const auto begin = range.first;
	const auto end = range.second;
	for (auto i = begin; i != end; ++i)
		out[i->r] = 1;
}
catch (std::out_of_range& e)
{
}


#endif // TEST_H

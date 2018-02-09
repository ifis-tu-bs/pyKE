#ifndef TEST_H
#	define TEST_H
#	include "Setting.h"
#	include "Reader.h"
#	include "Triple.h"
#	include <algorithm> // std::binary_search


extern "C" void query_head(char*, ent_id, rel_id);
extern "C" void query_tail(ent_id, char*, rel_id);
extern "C" void query_rel(ent_id, ent_id, char*);


extern "C" void query_head(
		char* out,
		ent_id tail,
		rel_id relation)
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
		ent_id head,
		char* out,
		rel_id relation)
try
{
	// assuming out has size `entityTotal` and is zero-initialized
	const auto range = std::equal_range(lefTail.at(head), rigTail.at(head),
			Triple(head, 0, relation), Triple::cmp_r);
	const auto begin = range.first;
	const auto end = range.second;
	for (auto i = begin; i != end; ++i)
		out[i->h] = 1;
}
catch (std::out_of_range& e)
{
}


extern "C" void query_rel(
		ent_id head,
		ent_id tail,
		char* out)
try
{
	// assuming out has size `relationTotal` and is zero-initialized
	const auto range = std::equal_range(lefRel.at(head), rigRel.at(head),
			Triple(head, tail, 0), Triple::cmp_t);
	const auto begin = range.first;
	const auto end = range.second;
	for (auto i = begin; i != end; ++i)
		out[i->h] = 1;
}
catch (std::out_of_range& e)
{
}


#endif // TEST_H

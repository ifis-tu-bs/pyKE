#ifndef READER_H
#	define READER_H
#	include "Setting.h"
#	include "Triple.h"
#	include <cstdlib>
#	include <vector> // std::vector
#	include <algorithm> // std::sort
#	include <fstream> // std::ifstream


// reordered trainList for lookup
// with precompiled lower and upper bounds for each element

// used when searching heads for (tail,rel) questions
// primary key is the tail
std::vector<Triple> trainHead;
std::vector<std::vector<Triple>::const_iterator> lefHead;
std::vector<std::vector<Triple>::const_iterator> rigHead;
// used for tail search on (head,rel) questions
// primary key is the head
std::vector<Triple> trainTail;
std::vector<std::vector<Triple>::const_iterator> lefTail;
std::vector<std::vector<Triple>::const_iterator> rigTail;
// used for rel search on (head,tail) questions
// primary key is the head
std::vector<Triple> trainRel;
std::vector<std::vector<Triple>::const_iterator> lefRel;
std::vector<std::vector<Triple>::const_iterator> rigRel;

// mean entity frequency for each relation
std::vector<float> meanh;
std::vector<float> meant;


extern "C"
int importTrainFiles(
		const char* inPath,
		ent_id entities,
		rel_id relations)
try
{
	entityTotal = entities;
	relationTotal = relations;

	std::vector<long int> freqr(relations);
	std::vector<long int> freqe(entities);
	{
		std::ifstream fin(inPath);
		size_t trainTotal;
		fin >> trainTotal;
		trainList.resize(trainTotal);
		for (auto& i: trainList)
		{
			fin >> i.h >> i.t >> i.r;
			++freqe.at(i.t);
			++freqe.at(i.h);
			++freqr.at(i.r);
		}
	}

	trainRel = trainTail = trainHead = trainList;
	std::sort(trainHead.begin(), trainHead.end(), Triple::cmp_trh);
	std::sort(trainTail.begin(), trainTail.end(), Triple::cmp_hrt);
	std::sort(trainRel.begin(), trainRel.end(), Triple::cmp_htr);

	lefHead.assign(entities, trainHead.cbegin());
	rigHead.assign(entities, trainHead.cbegin());
	{
		const auto end = trainHead.cend();
		auto i = trainHead.cbegin();
		ent_id last = i->t;
		for (++i; i != end; last = i->t, ++i)
		{
			if (i->t == last)
				continue;
			rigHead.at(last) = lefHead.at(i->t) = i;
		}
	}
	lefHead.at(trainHead.front().t) = trainHead.cbegin();
	rigHead.at(trainHead.back().t) = trainHead.cend();

	lefTail.assign(entities, trainTail.cbegin());
	rigTail.assign(entities, trainTail.cbegin());
	{
		const auto end = trainTail.cend();
		auto i = trainTail.cbegin();
		ent_id last = i->h;
		for (++i; i != end; last = i->h, ++i)
		{
			if (i->h == last)
				continue;
			rigTail.at(last) = lefTail.at(i->h) = i;
		}
	}
	lefTail.at(trainTail.front().h) = trainTail.cbegin();
	rigTail.at(trainTail.back().h) = trainTail.cend();

	lefRel.assign(entities, trainRel.cbegin());
	rigRel.assign(entities, trainRel.cbegin());
	{
		const auto end = trainRel.cend();
		auto i = trainRel.cbegin();
		ent_id last = i->h;
		for (++i; i != end; last = i->h, ++i)
		{
			if (i->h == last)
				continue;
			rigRel.at(last) = lefRel.at(i->h) = i;
		}
	}
	lefRel.at(trainRel.front().h) = trainRel.cbegin();
	rigRel.at(trainRel.back().h) = trainRel.cend();

	meanh.assign(relations, 0.);
	for (ent_id i = 0; i < entities; ++i)
	{
		const auto lower = lefHead[i];
		const auto upper = rigHead[i];
		if (lower >= upper)
			continue;
		for (auto j = lower + 1; j != upper; ++j)
			if (j->r != (j - 1)->r)
				meanh.at(j->r) += 1.;
	}
	for (rel_id i = 0; i < relations; ++i)
		meanh[i] = meanh[i] > .5 ? freqr[i] / meanh[i] : 0;

	meant.assign(relations, 0.);
	for (ent_id i = 0; i < entities; ++i)
	{
		const auto lower = lefTail[i];
		const auto upper = rigTail[i];
		if (lower >= upper)
			continue;
		for (auto j = lower + 1; j != upper; ++j)
			if (j->r != (j - 1)->r)
				meant.at(j->r) += 1.;
	}

	for (rel_id i = 0; i < relations; ++i)
		meant[i] = meant[i] > .5 ? freqr[i] / meant[i] : 0;
	return 0;
}
catch (std::out_of_range& e)
{
	return 1;
}


#endif // READER_H

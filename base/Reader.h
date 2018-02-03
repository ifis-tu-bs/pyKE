#ifndef READER_H
#	define READER_H
#	include "Setting.h"
#	include "Triple.h"
#	include <cstdlib>
#	include <algorithm>
#	include <fstream>


long int* lefHead;
long int* rigHead;
long int* lefTail;
long int* rigTail;
long int* lefRel;
long int* rigRel;
float* meanh;
float* meant;


Triple* trainList;
Triple* trainHead;
Triple* trainTail;
Triple* trainRel;
Triple* testList;
Triple* tripleList;


typedef int (*logCall)(int);


extern "C"
void importTrainFiles(
		const char* inPath,
		long int entities,
		long int relations,
		logCall log)
{
	relationTotal = relations;
	entityTotal = entities;
	log((int)relations*entities);

	std::ifstream fin(inPath);
	fin >> trainTotal;
	trainList = (Triple*) calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple*) calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple*) calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple*) calloc(trainTotal, sizeof(Triple));
	log((int)trainTotal);

	long int* freqr = (long int*) calloc(relationTotal, sizeof(long int));
	long int* freqe = (long int*) calloc(entityTotal, sizeof(long int));
	const auto* end = trainList + trainTotal;
	for (auto* i = trainList; i < end; ++i)
	{
		fin >> i->h >> i->t >> i->r;
		++freqe[i->t];
		++freqe[i->h];
		++freqr[i->r];
	}
	log(300);

	memcpy(trainHead, trainList, sizeof(Triple)*trainTotal);
	memcpy(trainTail, trainList, sizeof(Triple)*trainTotal);
	memcpy(trainRel, trainList, sizeof(Triple)*trainTotal);
	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_hrt);
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_trh);
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rht);
	log(500);

	lefHead = (long int *)calloc(entityTotal, sizeof(long int));
	rigHead = (long int *)calloc(entityTotal, sizeof(long int));
	lefTail = (long int *)calloc(entityTotal, sizeof(long int));
	rigTail = (long int *)calloc(entityTotal, sizeof(long int));
	lefRel = (long int *)calloc(entityTotal, sizeof(long int));
	rigRel = (long int *)calloc(entityTotal, sizeof(long int));
	log(600);

	memset(rigHead, -1, sizeof(long int)*entityTotal);
	memset(rigTail, -1, sizeof(long int)*entityTotal);
	memset(rigRel, -1, sizeof(long int)*entityTotal);
	for (long int i = 1; i < trainTotal; ++i)
	{
		if (trainTail[i].t != trainTail[i - 1].t)
		{
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h)
		{
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h)
		{
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;
	log(700);

	meanh = (float*)calloc(relationTotal, sizeof(float));
	meant = (float*)calloc(relationTotal, sizeof(float));
	for (long int i = 0; i < entityTotal; i++)
	{
		for (long int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				meanh[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			meanh[trainHead[lefHead[i]].r] += 1.0;
		for (long int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				meant[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			meant[trainTail[lefTail[i]].r] += 1.0;
	}
	for (long int i = 0; i < relationTotal; ++i)
	{
		meanh[i] = freqr[i] / meanh[i];
		meant[i] = freqr[i] / meant[i];
	}
	free(freqr);
	free(freqe);
	log(800);
}


extern "C"
void importTestFiles(
		const char* testname,
		const char* trainname,
		const char* validname)
{
	std::ifstream ftest(testname);
	std::ifstream ftrain(trainname);
	std::ifstream fvalid(validname);

	ftest >> testTotal;
	ftrain >> trainTotal;
	fvalid >> validTotal;
	tripleTotal = testTotal + trainTotal + validTotal;

	testList = (Triple*) calloc(testTotal, sizeof(Triple));
	tripleList = (Triple*) calloc(tripleTotal, sizeof(Triple));

	for (long int i = 0; i < testTotal; i++)
	{
		ftest >> testList[i].h;
		ftest >> testList[i].t;
		ftest >> testList[i].r;
		tripleList[i] = testList[i];
	}

	for (long int i = testTotal; i < trainTotal + testTotal; i++)
	{
		ftrain >> tripleList[i].h;
		ftrain >> tripleList[i].t;
		ftrain >> tripleList[i].r;
	}

	for (long int i = testTotal + trainTotal; i < tripleTotal; i++)
	{
		fvalid >> tripleList[i].h;
		fvalid >> tripleList[i].t;
		fvalid >> tripleList[i].r;
	}

	std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_hrt);
}


#endif // READER_H

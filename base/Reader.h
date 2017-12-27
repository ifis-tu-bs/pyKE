#ifndef READER_H
#	define READER_H
#	include "Setting.h"
#	include "Triple.h"
#	include <cstdlib>
#	include <algorithm>


extern "C" void importTrainFiles(void);
extern "C" void importTestFiles(void);


INT* freqRel;
INT* freqEnt;
INT* lefHead;
INT* rigHead;
INT* lefTail;
INT* rigTail;
INT* lefRel;
INT* rigRel;
REAL* left_mean;
REAL* right_mean;


Triple* trainList;
Triple* trainHead;
Triple* trainTail;
Triple* trainRel;


extern "C"
void importTrainFiles(void)
{

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal);
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);
	std::sort(trainList, trainList + trainTotal, Triple::cmp_hrt);
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	for (INT i = 1; i < tmp; i++)
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}

	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_hrt);
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_trh);
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rht);
	printf("The total of train triples is %ld.\n", trainTotal);

	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < trainTotal; ++i)
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

	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++)
	{
		for (INT j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; ++i)
	{
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}


Triple* testList;
Triple* tripleList;


extern "C"
void importTestFiles(void)
{
	INT tmp;

	FILE* frelation = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(frelation, "%ld", &relationTotal);
	fclose(frelation);

	FILE* fentity = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fentity, "%ld", &entityTotal);
	fclose(fentity);

	FILE* ftest = fopen((inPath + "test2id.txt").c_str(), "r");
	FILE* ftrain = fopen((inPath + "train2id.txt").c_str(), "r");
	FILE* fvalid = fopen((inPath + "valid2id.txt").c_str(), "r");
	tmp = fscanf(ftest, "%ld", &testTotal);
	tmp = fscanf(ftrain, "%ld", &trainTotal);
	tmp = fscanf(fvalid, "%ld", &validTotal);
	tripleTotal = testTotal + trainTotal + validTotal;
	testList = (Triple*) calloc(testTotal, sizeof(Triple));
	tripleList = (Triple*) calloc(tripleTotal, sizeof(Triple));

	for (INT i = 0; i < testTotal; i++)
	{
		tmp = fscanf(ftest, "%ld", &testList[i].h);
		tmp = fscanf(ftest, "%ld", &testList[i].t);
		tmp = fscanf(ftest, "%ld", &testList[i].r);
		tripleList[i] = testList[i];
	}
	fclose(ftest);

	for (INT i = 0; i < trainTotal; i++)
	{
		tmp = fscanf(ftrain, "%ld", &tripleList[i + testTotal].h);
		tmp = fscanf(ftrain, "%ld", &tripleList[i + testTotal].t);
		tmp = fscanf(ftrain, "%ld", &tripleList[i + testTotal].r);
	}
	fclose(ftrain);

	for (INT i = 0; i < validTotal; i++)
	{
		tmp = fscanf(fvalid, "%ld", &tripleList[i + testTotal + trainTotal].h);
		tmp = fscanf(fvalid, "%ld", &tripleList[i + testTotal + trainTotal].t);
		tmp = fscanf(fvalid, "%ld", &tripleList[i + testTotal + trainTotal].r);
	}
	fclose(fvalid);

	std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_hrt);
	printf("The total of test triples is %ld.\n", testTotal);
}
#endif // READER_H

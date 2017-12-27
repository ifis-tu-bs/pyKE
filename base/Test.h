#ifndef TEST_H
#	define TEST_H
#	include "Setting.h"
#	include "Reader.h"


static bool exists(INT, INT, INT);
extern "C" void getHeadBatch(INT*, INT*, INT*);
extern "C" void getTailBatch(INT*, INT*, INT*);
extern "C" void testHead(REAL*);
extern "C" void testTail(REAL*);
extern "C" void test(void);


static bool exists(INT h, INT t, INT r)
{
	INT lef = 0;
	INT rig = tripleTotal - 1;
	while (lef + 1 < rig)
	{
		INT mid = (lef + rig) >> 1;
		auto& T = tripleList[mid];
		if (T.h == h)
		{
			if (T.r == r)
			{
				if (T.t == t)
					return true;
				if (T.t < t)
					rig = mid;
				else
					lef = mid;
				break;
			}
			if (T.r < r)
				rig = mid;
			else
				lef = mid;
			break;
		}
		if (T.h < h)
			rig = mid;
		else
			lef = mid;
	}
	return false;
}


extern Triple* testList;
INT lasthead = 0;
INT lasttail = 0;


extern "C"
void getHeadBatch(INT* h, INT* t, INT* r)
{
	auto& T = testList[lasthead];
	for (INT i = 0; i < entityTotal; i++)
	{
		h[i] = i;
		t[i] = T.t;
		r[i] = T.r;
	}
}


extern "C"
void getTailBatch(INT* h, INT* t, INT* r)
{
	auto& T = testList[lasttail];
	for (INT i = 0; i < entityTotal; i++)
	{
		h[i] = T.h;
		t[i] = i;
		r[i] = T.r;
	}
}


REAL counth1 = 0, counth1f = 0, countt1 = 0, countt1f = 0;
REAL counth3 = 0, counth3f = 0, countt3 = 0, countt3f = 0;
REAL counth10 = 0, counth10f = 0, countt10 = 0, countt10f = 0;
REAL ranksumh = 0, ranksumhf = 0, ranksumt = 0, ranksumtf = 0;


extern "C"
void testHead(
		REAL* con)
{
	auto& T = testList[lasthead];
	REAL minimal = con[T.h];

	INT rank = 1;
	INT rankf = 1;
	for (INT j = 0; j <= entityTotal; ++j)
	{
		REAL value = con[j];
		if (j == T.h or value >= minimal)
			continue;
		++rank;
		if (not exists(j, T.t, T.r))
			++rankf;
	}

	if (rankf <= 10)
	{
		++counth10f;
		if (rank <= 10)
			++counth10;
		if (rankf <= 3)
		{
			++counth3f;
			if (rank <= 3)
				++counth3;
			if (rankf <= 1)
			{
				++counth1f;
				if (rank <= 1)
					++counth1;
			}
		}
	}

	ranksumhf += rankf;
	ranksumh += rank;

	++lasthead;
}


extern "C"
void testTail(
		REAL* con)
{
	auto& T = testList[lasttail];

	REAL minimal = con[T.t];
	INT rank = 1;
	INT rankf = 1;
	for (INT j = 0; j <= entityTotal; j++)
	{
		REAL value = con[j];
		if (j == T.t or value >= minimal)
			continue;
		++rank;
		if (not exists(T.h, j, T.r))
			++rankf;
	}

	if (rankf <= 10)
	{
		++countt10f;
		if (rank <= 10)
			++countt10;
		if (rankf <= 3)
		{
			++countt3f;
			if (rank <= 3)
				++countt3;
			if (rankf <= 1)
			{
				++countt1f;
				if (rank <= 1)
					++countt1;
			}
		}
	}

	ranksumtf += rankf;
	ranksumt += rank;

	++lasttail;
}


extern "C"
void test(void)
{
	const auto& t = testTotal;
	printf("\tsum\t#top 10\t#top 3\t#top\n");
	printf("left\t%f\t%f\t%f\t%f\n",
			ranksumh / t, counth10 / t, counth3 / t, counth1 / t);
	printf("left(filter)\t%f\t%f\t%f\t%f\n",
			ranksumhf / t, counth10f / t, counth3f / t, counth1f / t);
	printf("right\t%f\t%f\t%f\t%f\n",
			ranksumt / t, countt10 / t, countt3 / t, countt1 / t);
	printf("right(filter)\t%f\t%f\t%f\t%f\n",
			ranksumtf / t, countt10 / t, countt3f / t, countt1f / t);
}


#endif // TEST_H

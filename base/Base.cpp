#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <thread>
#include <forward_list>

extern "C"
void bernSampling(INT* h, INT* t, INT* r, REAL* y, INT size, INT ne, INT nr, INT workers);

extern "C"
void sampling(INT* h, INT* t, INT* r, REAL* y, INT size, INT ne, INT nr, INT workers);

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();


template <bool bernFlag>
void getBatch(INT id, INT* h, INT* t, INT* r, REAL* y,
		INT size, INT ne, INT nr, INT workers)
{
	INT k = size / workers + (size % workers ? 1 : 0);
	INT begin = id * k;
	INT end = std::min(begin + k, size);
	REAL prob = 500;
	for (INT batch = begin; batch < end; batch++)
	{
		INT i = rand_max(id, trainTotal);
		auto& T = trainList[i];
		h[batch] = T.h;
		t[batch] = T.t;
		r[batch] = T.r;
		y[batch] = 1;
		INT last = size;
		for (INT times = 0; times < ne; times ++)
		{
			INT j = batch + last;
			if (bernFlag)
				prob = 1000 * meant[T.r] / (meant[T.r] + meanh[T.r]);
			if (randd(id) % 1000 < prob)
			{
				h[j] = T.h;
				t[j] = corrupt_head(id, T.h, T.r);
			}
			else
			{
				h[j] = corrupt_tail(id, T.t, T.r);
				t[j] = T.t;
			}
			r[j] = T.r;
			y[j] = -1;
			last += size;
		}
		for (INT times = 0; times < nr; times++)
		{
			INT j = batch + last;
			h[j] = T.h;
			t[j] = T.t;
			r[j] = corrupt_rel(id, T.h, T.t);
			y[j] = -1;
			last += size;
		}
	}
}


extern "C"
void sampling(INT* h, INT* t, INT* r, REAL* y, INT size, INT ne, INT nr,
		INT workers)
{
	std::forward_list<std::thread> threads;
	for (int i = 0; i < workers; ++i)
		threads.push_front(std::thread(getBatch<false>, i, h, t, r, y,
				size, ne, nr, workers));
	for (auto& thread: threads)
		thread.join();
}


extern "C"
void bernSampling(INT* h, INT* t, INT* r, REAL* y, INT size, INT ne, INT nr,
		INT workers)
{
	std::forward_list<std::thread> threads;
	for (int i = 0; i < workers; ++i)
		threads.push_front(std::thread(getBatch<true>, i, h, t, r, y,
				size, ne, nr, workers));
	for (auto& thread: threads)
		thread.join();
}

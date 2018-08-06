#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <algorithm> // std::min
#include <thread> // std::thread
#include <forward_list> // std::forward_list

extern "C"
void bernSampling(uint64_t* h, uint64_t* t, uint64_t* r, float* y, uint64_t size, uint64_t ne, uint64_t nr, uint64_t workers);

extern "C"
void sampling(uint64_t* h, uint64_t* t, uint64_t* r, float* y, uint64_t size, uint64_t ne, uint64_t nr, uint64_t workers);

/**
  id: worker id
  h: head
  t: tail
  r: relation
  y: score (-1/1)
  size: batch size
  ne: negative entities
  nr: negative relations
  workers: number of workers
*/
template <bool bernFlag>
void getBatch(uint64_t id, uint64_t* h, uint64_t* t, uint64_t* r, float* y,
		uint64_t size, uint64_t ne, uint64_t nr, uint64_t workers)
{
	uint64_t k = size / workers + (size % workers ? 1 : 0);
	uint64_t begin = id * k;
	uint64_t end = std::min(begin + k, size);
	float prob = 500;
	for (uint64_t batch = begin; batch < end; batch++)
	{
		uint64_t i = rand_max(id, trainList.size());
		auto& T = trainList[i];
		h[batch] = T.h;
		t[batch] = T.t;
		r[batch] = T.r;
		y[batch] = 1;
		uint64_t last = size;
		for (uint64_t times = 0; times < ne; times ++)
		{
			uint64_t j = batch + last;
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
		for (uint64_t times = 0; times < nr; times++)
		{
			uint64_t j = batch + last;
			h[j] = T.h;
			t[j] = T.t;
			r[j] = corrupt_rel(id, T.h, T.t);
			y[j] = -1;
			last += size;
		}
	}
}


extern "C"
void sampling(uint64_t* h, uint64_t* t, uint64_t* r, float* y, uint64_t size, uint64_t ne, uint64_t nr,
		uint64_t workers)
{
	std::forward_list<std::thread> threads;
	for (uint64_t i = 0; i < workers; ++i)
		threads.push_front(std::thread(getBatch<false>, i, h, t, r, y,
				size, ne, nr, workers));
	for (auto& thread: threads)
		thread.join();
}


extern "C"
void bernSampling(uint64_t* h, uint64_t* t, uint64_t* r, float* y, uint64_t size, uint64_t ne, uint64_t nr,
		uint64_t workers)
{
	std::forward_list<std::thread> threads;
	for (uint64_t i = 0; i < workers; ++i)
		threads.push_front(std::thread(getBatch<true>, i, h, t, r, y,
				size, ne, nr, workers));
	for (auto& thread: threads)
		thread.join();
}

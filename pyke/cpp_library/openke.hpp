#ifndef header_openke
#	define header_openke
#	include "Triple.h"
#	ifdef __cplusplus
extern "C"
{
#	endif
#	if 0
}
#	endif

extern "C"
void bernSampling(uint64_t* h, uint64_t* t, uint64_t* r, float* y, uint64_t size, uint64_t ne, uint64_t nr, uint64_t workers);

extern "C"
void sampling(uint64_t* h, uint64_t* t, uint64_t* r, float* y, uint64_t size, uint64_t ne, uint64_t nr, uint64_t workers);

extern "C"
int importTrainFiles(const char* inPath, int64_t entities, int64_t relations);

extern "C"
void query_head(char*, int64_t, int64_t);

extern "C"
void query_tail(int64_t, char*, int64_t);

extern "C"
void query_rel(int64_t, int64_t, char*);

#	if 0
extern "C"
{
#	endif
#	ifdef __cplusplus
}
#	endif
#endif // header_openke

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
int importTrainFiles(const char* inPath, ent_id entities, rel_id relations);


extern "C"
void query_head(char*, ent_id, rel_id);
extern "C"
void query_tail(ent_id, char*, rel_id);
extern "C"
void query_rel(ent_id, ent_id, char*);


extern "C"
uint64_t getEntityTotal(void);

extern "C"
uint64_t getRelationTotal(void);

extern "C"
uint64_t getTripleTotal(void);

extern "C"
uint64_t getTrainTotal(void);

extern "C"
uint64_t getTestTotal(void);


#	if 0
extern "C"
{
#	endif
#	ifdef __cplusplus
}
#	endif
#endif // header_openke

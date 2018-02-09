#ifndef SETTING_H
#define SETTING_H
#include "Triple.h"
#include <vector> // std::vector

extern "C"
uint64_t getEntityTotal();

extern "C"
uint64_t getRelationTotal();

extern "C"
uint64_t getTripleTotal();

extern "C"
uint64_t getTrainTotal();

extern "C"
uint64_t getTestTotal();


uint64_t entityTotal = 0;
uint64_t relationTotal = 0;
std::vector<Triple> trainList;
std::vector<Triple> testList;

extern "C"
uint64_t getEntityTotal()
{
	return entityTotal;
}


extern "C"
uint64_t getRelationTotal()
{
	return relationTotal;
}


extern "C"
uint64_t getTripleTotal()
{
	return trainList.size() + testList.size();
}


extern "C"
uint64_t getTrainTotal()
{
	return trainList.size();
}


extern "C"
uint64_t getTestTotal()
{
	return testList.size();
}


extern "C"
uint64_t getValidTotal()
{
	return 0;
}
#endif
